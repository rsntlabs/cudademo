use naga::{
    back::wgsl,
    front::cuda,
    valid::{Capabilities, ValidationFlags, Validator},
};
use wasm_bindgen::prelude::*;

pub const REVISION: &str = "02f26c369996e6204d38069048be9051c2ea4e74";

/// This is the actual Naga CUDA frontend, compiled into the browser's WASM module.
#[wasm_bindgen]
pub fn translate(source: &str) -> Result<String, String> {
    if source.len() > 64 * 1024 {
        return Err("Keep CUDA source below 64 KiB.".into());
    }
    let module = cuda::parse_str(
        source,
        &cuda::Options {
            workgroup_size: [64, 1, 1],
        },
    )
    .map_err(|error| {
        let location = error.span.location(source);
        format!(
            "CUDA · line {}, column {}: {}",
            location.line_number, location.line_position, error.message
        )
    })?;
    let info = Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&module)
        .map_err(|error| error.emit_to_string(source))?;
    wgsl::write_string(&module, &info, wgsl::WriterFlags::EXPLICIT_TYPES)
        .map_err(|error| format!("WGSL generation: {error}"))
}

#[wasm_bindgen]
pub fn compiler_revision() -> String {
    REVISION.into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn particle_kernel_translates_and_round_trips() {
        for source in [
            include_str!("../web/shaders/particles.cu"),
            include_str!("../web/shaders/vortex.cu"),
            include_str!("../web/shaders/wave.cu"),
        ] {
            let output = translate(source).expect("CUDA translation");
            let module = naga::front::wgsl::parse_str(&output).expect("WGSL parses");
            Validator::new(ValidationFlags::all(), Capabilities::empty())
                .validate(&module)
                .expect("WGSL validates");
            assert_eq!(module.entry_points.len(), 1);
            assert_eq!(module.entry_points[0].name, "particles");
            assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
            let bindings: Vec<_> = module
                .global_variables
                .iter()
                .filter_map(|(_, v)| v.binding.as_ref().map(|b| (b.group, b.binding)))
                .collect();
            assert_eq!(bindings, [(0, 0), (0, 1)]);
        }
    }

    #[test]
    fn edits_change_the_generated_shader_and_errors_have_locations() {
        let source = include_str!("../web/shaders/particles.cu");
        assert_ne!(
            translate(source).unwrap(),
            translate(&source.replace("0.22f", "0.88f")).unwrap()
        );
        assert!(translate("__global__ void broken( {")
            .unwrap_err()
            .contains("line 1"));
        assert!(translate(&" ".repeat(65537))
            .unwrap_err()
            .contains("64 KiB"));
    }
}
