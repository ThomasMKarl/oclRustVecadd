use half::f16;

pub trait HasOpenclString {
    fn as_opencl_string() -> &'static str;
}

pub trait OpenclNum {}

impl OpenclNum for i8 {}
impl OpenclNum for i16 {}
impl OpenclNum for i32 {}
impl OpenclNum for i64 {}
impl OpenclNum for u8 {}
impl OpenclNum for u16 {}
impl OpenclNum for u32 {}
impl OpenclNum for u64 {}
impl OpenclNum for f16 {}
impl OpenclNum for f32 {}
impl OpenclNum for f64 {}

impl HasOpenclString for i8 {
    fn as_opencl_string() -> &'static str {
        "char"
    }
}

impl HasOpenclString for i16 {
    fn as_opencl_string() -> &'static str {
        "short"
    }
}

impl HasOpenclString for i32 {
    fn as_opencl_string() -> &'static str {
        "int"
    }
}

impl HasOpenclString for i64 {
    fn as_opencl_string() -> &'static str {
        "long"
    }
}

impl HasOpenclString for u8 {
    fn as_opencl_string() -> &'static str {
        "uchar"
    }
}

impl HasOpenclString for u16 {
    fn as_opencl_string() -> &'static str {
        "ushort"
    }
}

impl HasOpenclString for u32 {
    fn as_opencl_string() -> &'static str {
        "uint"
    }
}

impl HasOpenclString for u64 {
    fn as_opencl_string() -> &'static str {
        "ulong"
    }
}

impl HasOpenclString for f16 {
    fn as_opencl_string() -> &'static str {
        "half"
    }
}

impl HasOpenclString for f32 {
    fn as_opencl_string() -> &'static str {
        "float"
    }
}

impl HasOpenclString for f64 {
    fn as_opencl_string() -> &'static str {
        "double"
    }
}

impl HasOpenclString for bool {
    fn as_opencl_string() -> &'static str {
        "bool"
    }
}

impl HasOpenclString for char {
    fn as_opencl_string() -> &'static str {
        "char"
    }
}
