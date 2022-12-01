use opencl3::program;
use opencl3::kernel;
use opencl3::context;
use opencl3::command_queue;
use opencl3::device;
use opencl3::event;
use std::path::Path;
use std::fs;
use std::io::Read;

pub fn create_queue(
    context: &context::Context,
    device: &device::Device,
) -> Result<command_queue::CommandQueue, String> {
    unsafe {
        let queue = match command_queue::CommandQueue::create_with_properties(
            &context,
            device.id(),
            command_queue::CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
                | command_queue::CL_QUEUE_PROFILING_ENABLE,
            0,
        ) {
            Ok(queue) => queue,
            Err(error) => return Err(format!("not able to create command queue: {}", error)),
        };

        Ok(queue)
    }
}

pub fn create_and_build_from_sources(
    context: &context::Context,
    sources: &[&Path],
    options: &str,
) -> Result<Vec<program::Program>, String> {
    let mut programs: Vec<program::Program> = Vec::new();

    for source in sources {
        let sourcestr = match source.to_str().ok_or_else(|| 0) {
            Ok(sourcestr) => sourcestr,
            Err(error) => return Err(format!("no valid path given: {}", error)),
        };

        if !source.is_file() {
            return Err(format!(
                "source {} does not exist or is not a file",
                sourcestr
            ));
        }

        let content = match fs::read_to_string(sourcestr) {
            Ok(content) => content,
            Err(error) => {
                return Err(format!(
                    "error reading source file {}: {}",
                    sourcestr, error
                ))
            }
        };

        programs.push(
            match program::Program::create_and_build_from_source(&context, &content, &options) {
                Ok(program) => program,
                Err(error) => {
                    return Err(format!(
                        "error building source file {}: {}",
                        sourcestr, error
                    ))
                }
            },
        );
    }

    Ok(programs)
}

pub fn create_and_build_from_binaries(
    context: &context::Context,
    sources: &[&Path],
    options: &str,
) -> Result<Vec<program::Program>, String> {
    let mut programs: Vec<program::Program> = Vec::new();

    for source in sources {
        let sourcestr = match source.to_str().ok_or_else(|| 0) {
            Ok(sourcestr) => sourcestr,
            Err(error) => return Err(format!("no valid path given: {}", error)),
        };

        if !source.is_file() {
            return Err(format!(
                "source {} does not exist or is not a file",
                sourcestr
            ));
        }

        let mut file = match fs::File::open(sourcestr) {
            Ok(file) => file,
            Err(error) => {
                return Err(format!(
                    "error reading source file {}: {}",
                    sourcestr, error
                ))
            }
        };

        let mut buffer = Vec::<u8>::new();
        match file.read_to_end(&mut buffer) {
            Ok(_) => (),
            Err(error) => {
                return Err(format!(
                    "error reading source file {}: {}",
                    sourcestr, error
                ))
            }
        };

        programs.push(
            match program::Program::create_and_build_from_binary(&context, &[&buffer[..]], &options)
            {
                Ok(program) => program,
                Err(error) => {
                    return Err(format!(
                        "error building source file {}: {}",
                        sourcestr, error
                    ))
                }
            },
        );
    }

    Ok(programs)
}

pub fn execute_kernel(
    queue: &command_queue::CommandQueue,
    kernel: &kernel::Kernel,
    elements: usize,
    wait: &Vec<event::Event>,
) -> Result<event::Event, String> {
    let local_work_size = [256 as usize];
    let global_work_size = [get_global_work_size(256, elements) as usize];

    let mut cl_events: Vec<opencl3::types::cl_event> = Vec::new();
    for event in wait {
        cl_events.push(event.get());
    }

    unsafe {
        let event = match command_queue::enqueue_nd_range_kernel(
            queue.get(),
            kernel.get(),
            1,
            std::ptr::null(),
            global_work_size.as_ptr(),
            local_work_size.as_ptr(),
            cl_events.len() as u32,
            if !cl_events.is_empty() {
                cl_events.as_ptr()
            } else {
                std::ptr::null()
            },
        ) {
            Ok(event) => event,
            Err(error) => return Err(format!("error executing kernel: {}", error)),
        };

        Ok(event::Event::from(event))
    }
}

pub fn get_global_work_size(local: usize, elements: usize) -> usize {
    let mult = (elements + local - 1) / local;
    mult * local
}
