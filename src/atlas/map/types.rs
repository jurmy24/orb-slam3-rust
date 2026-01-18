//! Core ID types for the Atlas map structures.

/// Unique identifier for a KeyFrame within a Map.
///
/// KeyFrameIds are assigned sequentially when KeyFrames are created.
/// They serve as lightweight handles for cross-referencing without
/// needing Arc/Rc, which simplifies ownership and avoids cyclic references.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KeyFrameId(pub u64);

impl KeyFrameId {
    /// Create a new KeyFrameId with the given value.
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

impl std::fmt::Display for KeyFrameId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "KF{}", self.0)
    }
}

/// Unique identifier for a MapPoint within a Map.
///
/// MapPointIds are assigned sequentially when MapPoints are created.
/// A MapPoint represents a 3D landmark observed by one or more KeyFrames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MapPointId(pub u64);

impl MapPointId {
    /// Create a new MapPointId with the given value.
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

impl std::fmt::Display for MapPointId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MP{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyframe_id_equality() {
        let id1 = KeyFrameId::new(42);
        let id2 = KeyFrameId::new(42);
        let id3 = KeyFrameId::new(43);

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_mappoint_id_display() {
        let id = MapPointId::new(123);
        assert_eq!(format!("{}", id), "MP123");
    }

    #[test]
    fn test_id_as_hashmap_key() {
        use std::collections::HashMap;

        let mut map: HashMap<KeyFrameId, &str> = HashMap::new();
        map.insert(KeyFrameId::new(1), "first");
        map.insert(KeyFrameId::new(2), "second");

        assert_eq!(map.get(&KeyFrameId::new(1)), Some(&"first"));
        assert_eq!(map.get(&KeyFrameId::new(3)), None);
    }
}
