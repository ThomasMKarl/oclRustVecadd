use opencl3::context;
use opencl3::types::cl_mem_flags;
use opencl3::command_queue;
use opencl3::event;
use opencl3::memory;
use opencl3::memory::ClMem;
use std::ffi::c_void;

pub enum MemMode {
    Read,
    Write,
    ReadWrite,
}

pub fn create_buffer<T>(
    context: &context::Context,
    input: &mut Vec<T>,
    mode: MemMode,
) -> Result<memory::Buffer<T>, String> {
    let alloc_flag = memory::CL_MEM_USE_HOST_PTR;
    let mem_flag: cl_mem_flags;
    match mode {
        MemMode::Read => mem_flag = memory::CL_MEM_READ_ONLY,
        MemMode::Write => mem_flag = memory::CL_MEM_WRITE_ONLY,
        MemMode::ReadWrite => mem_flag = memory::CL_MEM_READ_WRITE,
    }

    unsafe {
        let data = input.as_mut_ptr() as *mut c_void;
        let buffer: memory::Buffer<T> =
            match memory::Buffer::create(context, mem_flag | alloc_flag, input.len(), data) {
                Ok(buffer) => buffer,
                Err(error) => return Err(format!("error creating buffer: {}", error)),
            };

        Ok(buffer)
    }
}

pub fn write_buffer<T: Copy>(
    queue: &command_queue::CommandQueue,
    buffer: &mut memory::Buffer<T>,
    input: &mut Vec<T>,
    wait: &Vec<event::Event>,
) -> Result<event::Event, String> {
    unsafe {
        let mut cl_events: Vec<opencl3::types::cl_event> = Vec::new();
        for event in wait {
            cl_events.push(event.get());
        }

        let event = match command_queue::enqueue_write_buffer(
            queue.get(),
            buffer.get_mut(),
            command_queue::CL_NON_BLOCKING,
            0,
            (input.len() * std::mem::size_of::<T>()) as usize,
            input.as_ptr() as *mut c_void,
            cl_events.len() as u32,
            if !cl_events.is_empty() {
                cl_events.as_ptr()
            } else {
                std::ptr::null()
            },
        ) {
            Ok(event) => event,
            Err(error) => return Err(format!("error writing buffer: {}", error)),
        };

        Ok(event::Event::from(event))
    }
}

pub fn read_buffer<T: Copy>(
    queue: &command_queue::CommandQueue,
    buffer: &mut memory::Buffer<T>,
    input: &mut Vec<T>,
    wait: &Vec<event::Event>,
) -> Result<event::Event, String> {
    let mut cl_events: Vec<opencl3::types::cl_event> = Vec::new();
    for event in wait {
        cl_events.push(event.get());
    }

    unsafe {
        let event = match command_queue::enqueue_read_buffer(
            queue.get(),
            buffer.get(),
            command_queue::CL_NON_BLOCKING,
            0,
            (input.len() * std::mem::size_of::<T>()) as usize,
            input.as_mut_ptr() as *mut c_void,
            cl_events.len() as u32,
            if !cl_events.is_empty() {
                cl_events.as_ptr()
            } else {
                std::ptr::null()
            },
        ) {
            Ok(event) => event,
            Err(error) => return Err(format!("error reading buffer: {}", error)),
        };

        Ok(event::Event::from(event))
    }
}
