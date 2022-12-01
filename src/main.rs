use opencl3::context;
use opencl3::memory;
use opencl3::memory::ClMem;
use opencl3::kernel;
use opencl3::command_queue;
use opencl3::program;
use opencl3::event;
use std::fs;
use std::io::Write;
use std::path::Path;
use num_traits::NumOps;

use log::error;
use log::info;
use log::{debug, LevelFilter};
use log4rs::append::console::ConsoleAppender;
use log4rs::config::{Appender, Root};
use log4rs::Config;

pub mod clvecadd;
pub mod test;

use crate::clvecadd::traits;
use crate::clvecadd::setup;
use crate::clvecadd::buffer;
use crate::clvecadd::exec;

pub fn create_binary(program: &program::Program, path_to_bin: String) -> Result<(), String> {
    let bins = match program.get_binaries() {
        Ok(bins) => bins,
        Err(error) => return Err(format!("not able to get binaries: {}", error)),
    };

    let bin = match bins.first().ok_or_else(|| 0) {
        Ok(bin) => bin,
        Err(error) => return Err(format!("no programs built: {}", error)),
    };

    let mut file = match fs::File::create(&path_to_bin) {
        Ok(file) => file,
        Err(error) => return Err(format!("not able to create {}: {}", path_to_bin, error)),
    };

    match file.write(bin) {
        Ok(_) => (),
        Err(error) => return Err(format!("not able to write {}: {}", path_to_bin, error)),
    };

    Ok(())
}

pub fn prepare_kernel_for_vecadd<T: traits::OpenclNum + traits::HasOpenclString>(
    context: &context::Context,
    buffer_a: &memory::Buffer<T>,
    buffer_b: &memory::Buffer<T>,
    buffer_c: &memory::Buffer<T>,
    elements: usize,
) -> Result<kernel::Kernel, String> {
    let kernel_name = String::from("addVectors");
    let path_to_bin = String::from("src/bin/vecadd/vecadd.bin");
    let path_to_source = String::from("src/opencl/vecadd/vecadd.cl");

    let sources = [Path::new(&path_to_source)];
    let binary = [Path::new(&path_to_bin)];

    let mut options = String::from("-cl-std=CL3.0 -w -D ARRAY_TYPE=");
    options.push_str(<T>::as_opencl_string());

    let progs = match exec::create_and_build_from_binaries(&context, &binary, &options) {
        Ok(progs) => progs,
        Err(error) => {
            debug!("{}", error);
            match exec::create_and_build_from_sources(&context, &sources, &options) {
                Ok(progs) => progs,
                Err(error) => return Err(error),
            }
        }
    };

    let prog = match progs.first().ok_or_else(|| 0) {
        Ok(prog) => prog,
        Err(error) => return Err(format!("no programs built: {}", error)),
    };

    match create_binary(prog, path_to_bin) {
        Ok(_) => (),
        Err(error) => {
            debug!("{}", error);
        }
    };

    let kernel = match kernel::Kernel::create(prog, &kernel_name) {
        Ok(kernel) => kernel,
        Err(error) => return Err(format!("not able to get kernel: {}", error)),
    };

    unsafe {
        let mut arg: u32 = 0;
        match kernel.set_arg(arg, &buffer_a.get()) {
            Ok(_) => (),
            Err(error) => {
                return Err(format!(
                    "error setting kernel argument {}: {}",
                    arg + 1,
                    error
                ))
            }
        }
        arg += 1;

        match kernel.set_arg(arg, &buffer_b.get()) {
            Ok(_) => (),
            Err(error) => {
                return Err(format!(
                    "error setting kernel argument {}: {}",
                    arg + 1,
                    error
                ))
            }
        }
        arg += 1;

        match kernel.set_arg(arg, &buffer_c.get()) {
            Ok(_) => (),
            Err(error) => {
                return Err(format!(
                    "error setting kernel argument {}: {}",
                    arg + 1,
                    error
                ))
            }
        }
        arg += 1;

        match kernel.set_arg(arg, &elements) {
            Ok(_) => (),
            Err(error) => {
                return Err(format!(
                    "error setting kernel argument {}: {}",
                    arg + 1,
                    error
                ))
            }
        }
    }

    Ok(kernel)
}

pub fn vecadd<T: Default + Copy + traits::OpenclNum + traits::HasOpenclString>(
    context: &context::Context,
    queue: &command_queue::CommandQueue,
    a: &mut Vec<T>,
    b: &mut Vec<T>,
) -> Result<Vec<T>, String> {
    let size = std::cmp::min(a.len(), b.len());
    let mut c: Vec<T> = Vec::new();
    c.resize_with(size, || <T>::default());

    let mut buffer_a: memory::Buffer<T> = match buffer::create_buffer(&context, a, buffer::MemMode::Read) {
        Ok(buffer_a) => buffer_a,
        Err(error) => return Err(error),
    };

    let mut buffer_b: memory::Buffer<T> = match buffer::create_buffer(&context, b, buffer::MemMode::Read) {
        Ok(buffer_b) => buffer_b,
        Err(error) => return Err(error),
    };

    let mut buffer_c: memory::Buffer<T> = match buffer::create_buffer(&context, &mut c, buffer::MemMode::Write) {
        Ok(buffer_c) => buffer_c,
        Err(error) => return Err(error),
    };

    let kernel = match prepare_kernel_for_vecadd(&context, &buffer_a, &buffer_b, &buffer_c, size) {
        Ok(kernel) => kernel,
        Err(error) => return Err(error),
    };

    let mut wait_list: Vec<event::Event> = Vec::new();
    let write_a_event = match buffer::write_buffer(&queue, &mut buffer_a, a, &wait_list) {
        Ok(write_a_event) => write_a_event,
        Err(error) => return Err(error),
    };

    let write_b_event = match buffer::write_buffer(&queue, &mut buffer_b, b, &wait_list) {
        Ok(write_b_event) => write_b_event,
        Err(error) => return Err(error),
    };

    wait_list.push(write_a_event);
    wait_list.push(write_b_event);
    let execute_event = match exec::execute_kernel(&queue, &kernel, size, &wait_list) {
        Ok(execute_event) => execute_event,
        Err(error) => return Err(error),
    };
    wait_list.clear();

    wait_list.push(execute_event);
    let _read_event = match buffer::read_buffer(&queue, &mut buffer_c, &mut c, &wait_list) {
        Ok(read_event) => read_event,
        Err(error) => return Err(error),
    };

    match queue.finish() {
        Ok(_) => (),
        Err(error) => return Err(format!("cannot finish queue: {}", error)),
    }

    Ok(c)
}

pub fn vecadd_cpu<T: Copy + Default + NumOps>(a: &Vec<T>, b: &Vec<T>) -> Result<Vec<T>, String> {
    let size = std::cmp::min(a.len(), b.len());
    let mut c: Vec<T> = Vec::new();
    c.resize_with(size, || <T>::default());

    c = (0..size).map(|i| a[i] + b[i]).collect();

    Ok(c)
}

fn main_impl() -> Result<(), String> {
    let devices = match setup::get_all_gpus() {
        Ok(devices) => devices,
        Err(error) => return Err(error),
    };

    let (ctx, queue) = match setup::get_context_for_device(devices, 0) {
        Ok((ctx, queue)) => (ctx, queue),
        Err(error) => return Err(error),
    };

    let mut a: Vec<i32> = vec![1, 2, 3, 4, 5];
    let mut b: Vec<i32> = vec![1, 2, 3, 4, 5];
    let _c = match vecadd(&ctx, &queue, &mut a, &mut b) {
        Ok(c) => c,
        Err(error) => {
            info!("not able to perform vector addition on gpu, falling back to cpu...");
            debug!("{}", error);
            match vecadd_cpu(&a, &b) {
                Ok(c) => c,
                Err(error) => return Err(error),
            }
        }
    };

    Ok(())
}

fn main() -> Result<(), i32> {
    let stdout = ConsoleAppender::builder().build();
    let config = Config::builder()
        .appender(Appender::builder().build("stdout", Box::new(stdout)))
        .build(Root::builder().appender("stdout").build(LevelFilter::Trace))
        .unwrap();
    let _handle = log4rs::init_config(config).unwrap();

    match main_impl() {
        Ok(_) => (),
        Err(error) => {
            error!("internal error");
            if !error.is_empty() {
                debug!("{}", error);
            }
            return Err(-1);
        }
    };

    Ok(())
}
