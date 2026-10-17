pub mod decode;
pub mod encode;
pub mod splice;
pub mod wordarena;

pub const REGEX_GPT_2: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+";
