//! gpu.rs - GPUs via Vulkan. The loader is opened at run time, so a machine without Vulkan still gets the rest of the
//! report.

use ash::vk;
use serde::Serialize;
use std::collections::HashSet;
use std::ffi::CStr;

#[derive(Debug, Clone, Serialize)]
pub struct Heap {
    pub index: u32,
    pub size_bytes: u64,
    pub device_local: bool,
    /// Union of property flags of the memory types that live in this heap.
    pub type_flags: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Gpu {
    pub name: String,
    pub vendor: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub kind: &'static str,
    pub vulkan_version: String,
    pub driver_name: Option<String>,
    pub driver_info: Option<String>,
    pub heaps: Vec<Heap>,
    /// Largest DEVICE_LOCAL heap (VRAM on a discrete card, the shared carve-out on an iGPU).
    pub device_local_bytes: u64,
    /// Largest heap reachable through a DEVICE_LOCAL|HOST_VISIBLE memory type.
    pub host_visible_device_local_bytes: u64,
    /// Discrete GPUs: true if the CPU can map (nearly) all of VRAM. None for integrated/UMA.
    pub resizable_bar: Option<bool>,
    pub external_memory_host: bool,
    pub min_imported_host_pointer_alignment: Option<u64>,
    pub subgroup_size: Option<u32>,
    pub subgroup_size_min: Option<u32>,
    pub subgroup_size_max: Option<u32>,
    pub notable_extensions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct GpuInfo {
    pub loader_version: Option<String>,
    pub gpus: Vec<Gpu>,
    pub error: Option<String>,
}

const NOTABLE: &[&str] = &[
    "VK_EXT_external_memory_host",
    "VK_EXT_memory_budget",
    "VK_EXT_subgroup_size_control",
    "VK_KHR_cooperative_matrix",
    "VK_NV_cooperative_matrix2",
    "VK_KHR_shader_integer_dot_product",
    "VK_KHR_shader_float16_int8",
    "VK_KHR_16bit_storage",
    "VK_KHR_8bit_storage",
    "VK_KHR_shader_bfloat16",
    "VK_EXT_shader_atomic_float",
    "VK_KHR_buffer_device_address",
    "VK_KHR_timeline_semaphore",
];

fn ver(v: u32) -> String {
    format!("{}.{}.{}", vk::api_version_major(v), vk::api_version_minor(v), vk::api_version_patch(v))
}

fn vendor_name(id: u32) -> String {
    match id {
        0x10DE => "NVIDIA",
        0x1002 | 0x1022 => "AMD",
        0x8086 => "Intel",
        0x5143 => "Qualcomm",
        0x13B5 => "ARM",
        0x1414 => "Microsoft",
        0x10005 => "Mesa",
        _ => "other",
    }
    .to_string()
}

fn kind(t: vk::PhysicalDeviceType) -> &'static str {
    match t {
        vk::PhysicalDeviceType::DISCRETE_GPU => "discrete",
        vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual",
        vk::PhysicalDeviceType::CPU => "cpu",
        _ => "other",
    }
}

fn mem_flag_names(f: vk::MemoryPropertyFlags) -> Vec<&'static str> {
    let mut v = Vec::new();
    for (bit, name) in [
        (vk::MemoryPropertyFlags::DEVICE_LOCAL, "DEVICE_LOCAL"),
        (vk::MemoryPropertyFlags::HOST_VISIBLE, "HOST_VISIBLE"),
        (vk::MemoryPropertyFlags::HOST_COHERENT, "HOST_COHERENT"),
        (vk::MemoryPropertyFlags::HOST_CACHED, "HOST_CACHED"),
        (vk::MemoryPropertyFlags::LAZILY_ALLOCATED, "LAZILY_ALLOCATED"),
        (vk::MemoryPropertyFlags::PROTECTED, "PROTECTED"),
        (vk::MemoryPropertyFlags::DEVICE_COHERENT_AMD, "DEVICE_COHERENT_AMD"),
        (vk::MemoryPropertyFlags::DEVICE_UNCACHED_AMD, "DEVICE_UNCACHED_AMD"),
    ] {
        if f.contains(bit) {
            v.push(name);
        }
    }
    v
}

fn cstr(r: Result<&CStr, std::ffi::FromBytesUntilNulError>) -> Option<String> {
    r.ok().map(|s| s.to_string_lossy().trim().to_string()).filter(|s| !s.is_empty())
}

pub fn collect() -> GpuInfo {
    // SAFETY: loads the system Vulkan loader; ash checks the symbols it resolves.
    let entry = match unsafe { ash::Entry::load() } {
        Ok(e) => e,
        Err(e) => return GpuInfo { error: Some(format!("no Vulkan loader: {e}")), ..Default::default() },
    };
    // SAFETY: entry is valid.
    let inst_ver = unsafe { entry.try_enumerate_instance_version() }.ok().flatten().unwrap_or(vk::API_VERSION_1_0);
    let api = inst_ver.min(vk::API_VERSION_1_3);
    let app = vk::ApplicationInfo::default().application_name(c"hwscan").api_version(api);
    let ci = vk::InstanceCreateInfo::default().application_info(&app);
    // SAFETY: create info is valid; instance destroyed below.
    let instance = match unsafe { entry.create_instance(&ci, None) } {
        Ok(i) => i,
        Err(e) => {
            return GpuInfo {
                loader_version: Some(ver(inst_ver)),
                error: Some(format!("vkCreateInstance: {e}")),
                ..Default::default()
            };
        }
    };
    let mut info = GpuInfo { loader_version: Some(ver(inst_ver)), ..Default::default() };
    // SAFETY: instance is valid.
    match unsafe { instance.enumerate_physical_devices() } {
        Ok(pds) => {
            for pd in pds {
                info.gpus.push(describe(&instance, pd, api));
            }
        }
        Err(e) => info.error = Some(format!("vkEnumeratePhysicalDevices: {e}")),
    }
    // SAFETY: no child objects were created.
    unsafe { instance.destroy_instance(None) };
    info
}

fn describe(instance: &ash::Instance, pd: vk::PhysicalDevice, inst_api: u32) -> Gpu {
    // SAFETY (all calls in this fn): `pd` came from this instance, which outlives the call.
    let props = unsafe { instance.get_physical_device_properties(pd) };
    let exts: HashSet<String> = unsafe { instance.enumerate_device_extension_properties(pd) }
        .unwrap_or_default()
        .iter()
        .filter_map(|e| cstr(e.extension_name_as_c_str()))
        .collect();
    let dev_api = props.api_version;
    let has = |n: &str| exts.contains(n);
    let ext_host = has("VK_EXT_external_memory_host");

    let mut subgroup_size = None;
    let (mut sg_min, mut sg_max) = (None, None);
    let (mut driver_name, mut driver_info) = (None, None);
    let mut host_align = None;
    if inst_api >= vk::API_VERSION_1_1 && dev_api >= vk::API_VERSION_1_1 {
        let mut sg = vk::PhysicalDeviceSubgroupProperties::default();
        let mut ssc = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
        let mut drv = vk::PhysicalDeviceDriverProperties::default();
        let mut emh = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
        let want_ssc = dev_api >= vk::API_VERSION_1_3 || has("VK_EXT_subgroup_size_control");
        let want_drv = dev_api >= vk::API_VERSION_1_2 || has("VK_KHR_driver_properties");
        {
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sg);
            if want_ssc {
                p2 = p2.push_next(&mut ssc);
            }
            if want_drv {
                p2 = p2.push_next(&mut drv);
            }
            if ext_host {
                p2 = p2.push_next(&mut emh);
            }
            unsafe { instance.get_physical_device_properties2(pd, &mut p2) };
        }
        subgroup_size = Some(sg.subgroup_size);
        if want_ssc {
            sg_min = Some(ssc.min_subgroup_size);
            sg_max = Some(ssc.max_subgroup_size);
        }
        if want_drv {
            driver_name = cstr(drv.driver_name_as_c_str());
            driver_info = cstr(drv.driver_info_as_c_str());
        }
        if ext_host {
            host_align = Some(emh.min_imported_host_pointer_alignment);
        }
    }

    let mp = unsafe { instance.get_physical_device_memory_properties(pd) };
    let types = mp.memory_types_as_slice();
    let heaps: Vec<Heap> = mp
        .memory_heaps_as_slice()
        .iter()
        .enumerate()
        .map(|(i, h)| {
            let union = types
                .iter()
                .filter(|t| t.heap_index as usize == i)
                .fold(vk::MemoryPropertyFlags::empty(), |a, t| a | t.property_flags);
            Heap {
                index: i as u32,
                size_bytes: h.size,
                device_local: h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL),
                type_flags: mem_flag_names(union).into_iter().map(String::from).collect(),
            }
        })
        .collect();
    let device_local_bytes = heaps.iter().filter(|h| h.device_local).map(|h| h.size_bytes).max().unwrap_or(0);
    let dl_hv = vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE;
    let host_visible_device_local_bytes = types
        .iter()
        .filter(|t| t.property_flags.contains(dl_hv))
        .filter_map(|t| heaps.get(t.heap_index as usize).map(|h| h.size_bytes))
        .max()
        .unwrap_or(0);
    let k = kind(props.device_type);
    let resizable_bar = (k == "discrete" && device_local_bytes > 0)
        .then_some(host_visible_device_local_bytes as f64 >= 0.9 * device_local_bytes as f64);

    let mut notable: Vec<String> = NOTABLE.iter().filter(|n| has(n)).map(|n| n.to_string()).collect();
    notable.sort();

    Gpu {
        name: cstr(props.device_name_as_c_str()).unwrap_or_else(|| "?".into()),
        vendor: vendor_name(props.vendor_id),
        vendor_id: props.vendor_id,
        device_id: props.device_id,
        kind: k,
        vulkan_version: ver(dev_api),
        driver_name,
        driver_info,
        heaps,
        device_local_bytes,
        host_visible_device_local_bytes,
        resizable_bar,
        external_memory_host: ext_host,
        min_imported_host_pointer_alignment: host_align,
        subgroup_size,
        subgroup_size_min: sg_min,
        subgroup_size_max: sg_max,
        notable_extensions: notable,
    }
}
