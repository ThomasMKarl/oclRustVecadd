use opencl3::platform;
use opencl3::command_queue;
use opencl3::context;
use opencl3::device;
use log::info;

use crate::clvecadd::exec;

pub enum DeviceType {
    All,
    Gpu,
    Acc,
    Cpu,
    Custom,
    Default,
}

pub fn log_platform_info(platform: &platform::Platform) {
    match platform.name() {
        Ok(value) => info!("Platform name: {}", value),
        Err(error) => panic!("No platform name: {:?}", error),
    }

    match platform.profile() {
        Ok(value) => info!("Profile: {}", value),
        Err(error) => panic!("No profile: {:?}", error),
    }

    match platform.version() {
        Ok(value) => info!("Version: {}", value),
        Err(error) => panic!("No version: {:?}", error),
    }

    match platform.vendor() {
        Ok(value) => info!("Vendor: {}", value),
        Err(error) => panic!("No vendor: {:?}", error),
    }

    match platform.extensions() {
        Ok(value) => info!("Extensions: {}", value),
        Err(error) => panic!("No extensions: {:?}", error),
    }
}

pub fn append_devices_from_platform(
    devices: &mut Vec<device::Device>,
    platform: &platform::Platform,
    dtype: DeviceType,
) -> Result<(), String> {
    let t: u64;
    match dtype {
        DeviceType::All => t = opencl3::device::CL_DEVICE_TYPE_ALL,
        DeviceType::Gpu => t = opencl3::device::CL_DEVICE_TYPE_GPU,
        DeviceType::Acc => t = opencl3::device::CL_DEVICE_TYPE_ACCELERATOR,
        DeviceType::Cpu => t = opencl3::device::CL_DEVICE_TYPE_CPU,
        DeviceType::Custom => t = opencl3::device::CL_DEVICE_TYPE_CUSTOM,
        DeviceType::Default => t = opencl3::device::CL_DEVICE_TYPE_DEFAULT,
    }

    let device_ids = match platform.get_devices(t) {
        Ok(device_ids) => device_ids,
        Err(error) => return Err(format!("error getting device ids: {}", error)),
    };

    for device_id in device_ids {
        devices.push(device::Device::new(device_id));
    }

    Ok(())
}

pub fn get_all_gpus() -> Result<Vec<device::Device>, String> {
    let platforms = match platform::get_platforms() {
        Ok(platforms) => platforms,
        Err(error) => return Err(format!("error getting platforms: {}", error)),
    };

    let mut devices: Vec<device::Device> = Vec::new();
    for platform in platforms {
        log_platform_info(&platform);
        match append_devices_from_platform(&mut devices, &platform, DeviceType::Gpu) {
            Ok(_) => (),
            Err(error) => return Err(error),
        };
    }

    Ok(devices)
}

pub fn get_context_for_device(
    devices: Vec<device::Device>,
    device_number: usize,
) -> Result<(context::Context, command_queue::CommandQueue), String> {
    let device = match devices.get(device_number).ok_or_else(|| 0) {
        Ok(device) => device,
        Err(error) => {
            return Err(format!(
                "requesting device number {}, but only {} exist: {}",
                device_number,
                devices.len(),
                error
            ))
        }
    };

    let context: context::Context = match context::Context::from_device(device) {
        Ok(context) => context,
        Err(error) => return Err(format!("error getting context: {}", error)),
    };

    let queue = match exec::create_queue(&context, &device) {
        Ok(queue) => queue,
        Err(error) => return Err(error),
    };

    Ok((context, queue))
}
