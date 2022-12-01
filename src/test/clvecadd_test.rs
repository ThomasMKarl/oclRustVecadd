#[cfg(test)]
mod clvecadd_test {
    #[test]
    fn gpu_found() -> Result<(), String> {
        match crate::clvecadd::setup::get_all_gpus() {
            Ok(_) => return Ok(()),
            Err(error) => return Err(error),
        };
    }

    #[test]
    fn get_context() -> Result<(), String> {
        let devices: Vec<opencl3::device::Device>;
        unsafe {
            devices = crate::clvecadd::setup::get_all_gpus().unwrap_unchecked();
        }
        match crate::clvecadd::setup::get_context_for_device(devices, 0) {
            Ok(_) => return Ok(()),
            Err(error) => return Err(error),
        };
    }

    #[test]
    fn perform_vecadd_on_cpu() -> Result<(), String> {
        let a: Vec<i32> = vec![1, 2, 3, 4, 5];
        let b: Vec<i32> = vec![6, 7, 8, 9, 10, 11];
        let desired_outcome: Vec<i32> = vec![7, 9, 11, 13, 15];

        match crate::vecadd_cpu(&a, &b) {
            Ok(c) => {
                assert_eq!(c, desired_outcome);
                return Ok(());
            },
            Err(error) => return Err(error),
        }
    }

    #[test]
    fn perform_vecadd_on_gpu() -> Result<(), String> {
        let devices: Vec<opencl3::device::Device>;
        let ctx: opencl3::context::Context;
        let queue: opencl3::command_queue::CommandQueue;
        unsafe {
            devices = crate::clvecadd::setup::get_all_gpus().unwrap_unchecked();
            (ctx, queue) = crate::clvecadd::setup::get_context_for_device(devices, 0).unwrap_unchecked();
        }

        let mut a: Vec<i32> = vec![1, 2, 3, 4, 5];
        let mut b: Vec<i32> = vec![6, 7, 8, 9, 10, 11];
        let desired_outcome: Vec<i32> = vec![7, 9, 11, 13, 15];
        match crate::vecadd(&ctx, &queue, &mut a, &mut b) {
            Ok(c) => {
                assert_eq!(c, desired_outcome);
                return Ok(());
            },
            Err(error) => return Err(error),
        };
    }

    #[test]
    fn perform_vecadd_with_fallback() -> Result<(), String> {
        let devices: Vec<opencl3::device::Device>;
        let ctx: opencl3::context::Context;
        let queue: opencl3::command_queue::CommandQueue;
        unsafe {
            devices = crate::clvecadd::setup::get_all_gpus().unwrap_unchecked();
            (ctx, queue) = crate::clvecadd::setup::get_context_for_device(devices, 0).unwrap_unchecked();
        }

        let mut a: Vec<i32> = vec![1, 2, 3, 4, 5];
        let mut b: Vec<i32> = vec![6, 7, 8, 9, 10, 11];
        let desired_outcome: Vec<i32> = vec![7, 9, 11, 13, 15];
        match crate::vecadd(&ctx, &queue, &mut a, &mut b) {
            Err(error) => return Err(error),
            Ok(_) => {
                match crate::vecadd(&ctx, &queue, &mut a, &mut b) {
                    Ok(c) => {
                        assert_eq!(c, desired_outcome);
                        return Ok(());
                    },
                    Err(error) => return Err(error),
                };
            }
        };
    }
}
