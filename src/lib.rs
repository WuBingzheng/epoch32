//! We generally use the 64-bit type to represent second-level UNIX timestamps.
//! This provides an extremely large range, exceeding 29.2 billion years.
//! This is a waste in scenarios sensitive to memory usage. Using a 32-bit type,
//! however, allows for a range of 136 years, which is sufficient for most
//! applications. Yet, considering UNIX timestamps start from 1970, this range
//! spans from 1970 to 2106. The year 2106 doesn't seem very distant,
//! potentially causing overflow issues.
//!
//! If we can specify a starting time, such as the release time of the
//! application, we can bypass the limitation of 1970. For example, if an
//! application releases in 2024 and we only need timestamps from 2024 onwards,
//! then by using 2024 as the starting point, we can represent a range from
//! 2024 to 2160. Here 2160 seems much more distant and safe.
//!
//! This crate provides such a simple encapsulation by specifying a starting
//! year and using a 32-bit type to represent timestamps.
//!
//! As time passes, the value of this crate will become increasingly evident.
//!
//! # Examples
//! 
//! ```
//! use epoch32::Epoch32;
//! type MyEpoch = Epoch32<2024>; // 2024-2160
//!
//! assert_eq!(std::mem::size_of::<MyEpoch>(), 4); // 32-bits
//!
//! println!("{}", MyEpoch::now()); // get current time, and show as UNIX Epoch
//!
//! let ts1 = MyEpoch::try_from_unix_epoch(1719403339).unwrap(); // convert from UNIX Epoch
//! assert_eq!(ts1.to_unix_epoch(), 1719403339); // convert to UNIX Epoch
//!
//! let ts2 = MyEpoch::try_from_unix_epoch(1719403349).unwrap();
//! let tensecs = std::time::Duration::from_secs(10);
//! assert!(ts1 < ts2); // compare
//! assert_eq!(ts1 + tensecs, ts2); // add duration
//! assert_eq!(ts2.duration_since(ts1), Some(tensecs)); // calulate diff duration
//! println!("{} {}", ts1, ts2);
//! ```
//!
//! # Features
//!
//! - `serde` enables serde traits integration (`Serialize`/`Deserialize`)
//!

use std::fmt;
use std::ops::{Add, AddAssign, Sub, SubAssign};
use std::time::{SystemTime, Duration, UNIX_EPOCH};

#[derive(Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Hash, Default, Debug)]
pub struct Epoch32<const Y: u32>(u32);

impl<const Y: u32> Epoch32<Y> {
    const DIFF_SECS: u64 = (Y as u64 - 1970) * 365 * 86400;

    fn try_from_raw_u64(n: u64) -> Option<Self> {
        match u32::try_from(n) {
            Ok(n) => Some(Epoch32(n)),
            Err(_) => None,
        }
    }

    fn try_from_sub_u64(n1: u64, n2: u64) -> Option<Self> {
        if n1 >= n2 {
            Self::try_from_raw_u64(n1 - n2)
        } else {
            None
        }
    }

    /// Try to convert from UNIX Epoch in `u64` type.
    /// 
    /// Return `None` if the input UNIX Epoch is earlier than the
    /// begin year or overflow for `u32`.
    /// 
    /// # Examples
    /// 
    /// ```
    /// use epoch32::Epoch32;
    /// type MyEpoch = Epoch32<2024>;
    /// assert_eq!(MyEpoch::try_from_unix_epoch(1234567890), None); // too early
    /// assert_eq!(MyEpoch::try_from_unix_epoch(12345678901234567), None); // overflow
    ///
    /// let ts = MyEpoch::try_from_unix_epoch(1719403339).unwrap();
    /// println!("{}", ts);
    /// ```
    pub fn try_from_unix_epoch(u: u64) -> Option<Self> {
        Self::try_from_sub_u64(u, Self::DIFF_SECS)
    }

    /// Convert to UNIX Epoch in `u64` type.
    /// 
    /// # Examples
    /// 
    /// ```
    /// use epoch32::Epoch32;
    /// type MyEpoch = Epoch32<2024>;
    ///
    /// let ts = MyEpoch::try_from_unix_epoch(1719403339).unwrap();
    /// assert_eq!(MyEpoch::to_unix_epoch(ts), 1719403339);
    /// ```
    pub fn to_unix_epoch(self) -> u64 {
        self.0 as u64 + Self::DIFF_SECS
    }

    /// Return current epoch.
    ///
    /// # Panics
    ///
    /// This function may panic if current time is earlier than the
    /// begin year or overflow for `u32`.
    pub fn now() -> Self {
        Self::try_from(&SystemTime::now()).expect("too big timestamp")
    }

    /// Returns the amount of time elapsed from an earlier point in time.
    ///
    /// # Examples
    /// 
    /// ```
    /// use epoch32::Epoch32;
    /// use std::time::Duration;
    /// type MyEpoch = Epoch32<2024>;
    ///
    /// let ts1 = MyEpoch::try_from_unix_epoch(1719403339).unwrap();
    /// let ts2 = MyEpoch::try_from_unix_epoch(1719403349).unwrap();
    /// assert_eq!(ts2.duration_since(ts1), Some(Duration::from_secs(10)));
    /// assert_eq!(ts1.duration_since(ts2), None);
    /// ```
    pub fn duration_since(self, earlier: Self) -> Option<Duration> {
        if self.0 >= earlier.0 {
            Some(Duration::from_secs((self.0 - earlier.0) as u64))
        } else {
            None
        }
    }

    /// Returns the amount of time elapsed since this instant.
    pub fn elapsed(self) -> Option<Duration> {
        Self::now().duration_since(self)
    }

    /// Add some duration.
    ///
    /// # Examples
    /// 
    /// ```
    /// use epoch32::Epoch32;
    /// use std::time::Duration;
    /// type MyEpoch = Epoch32<2024>;
    ///
    /// let ts1 = MyEpoch::try_from_unix_epoch(1719403339).unwrap();
    /// let ts2 = MyEpoch::try_from_unix_epoch(1719403349).unwrap();
    /// let dur = Duration::from_secs(10);
    /// assert_eq!(ts1.checked_add(dur), Some(ts2));
    ///
    /// let very_long = Duration::from_secs(9999999999999);
    /// assert_eq!(ts1.checked_add(very_long), None);
    /// ```
    pub fn checked_add(self, dur: Duration) -> Option<Self> {
        Self::try_from_raw_u64(self.0 as u64 + dur.as_secs())
    }

    /// Sub some duration.
    ///
    /// # Examples
    /// 
    /// ```
    /// use epoch32::Epoch32;
    /// use std::time::Duration;
    /// type MyEpoch = Epoch32<2024>;
    ///
    /// let ts1 = MyEpoch::try_from_unix_epoch(1719403339).unwrap();
    /// let ts2 = MyEpoch::try_from_unix_epoch(1719403349).unwrap();
    /// let dur = Duration::from_secs(10);
    /// assert_eq!(ts2.checked_sub(dur), Some(ts1));
    ///
    /// let very_long = Duration::from_secs(9999999999999);
    /// assert_eq!(ts2.checked_sub(very_long), None);
    /// ```
    pub fn checked_sub(self, dur: Duration) -> Option<Self> {
        Self::try_from_sub_u64(self.0 as u64, dur.as_secs())
    }
}

impl<const Y: u32> fmt::Display for Epoch32<Y> {
    /// Show as UNIX Epoch in `u64`.
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "{}", self.to_unix_epoch())
    }
}

impl<const Y: u32> TryFrom<&SystemTime> for Epoch32<Y> {
    type Error = ();

    fn try_from(st: &SystemTime) -> Result<Self, Self::Error> {
        match st.duration_since(UNIX_EPOCH) {
            Ok(n) => Self::try_from_unix_epoch(n.as_secs()).ok_or(()),
            Err(_) => Err(()),
        }
    }
}

impl<const Y: u32> Into<SystemTime> for Epoch32<Y> {
    fn into(self) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(self.to_unix_epoch())
    }
}

impl<const Y: u32> Add<Duration> for Epoch32<Y> {
    type Output = Self;

    /// # Panics
    ///
    /// This function may panic if the resulting point in time cannot be represented by the
    /// underlying data structure. See [`Epoch32::checked_add`] for a version without panic.
    fn add(self, dur: Duration) -> Self {
        self.checked_add(dur).expect("overflow when adding duration to instant")
    }
}

impl<const Y: u32> Sub<Duration> for Epoch32<Y> {
    type Output = Self;

    /// # Panics
    ///
    /// This function may panic if the resulting point in time cannot be represented by the
    /// underlying data structure. See [`Epoch32::checked_sub`] for a version without panic.
    fn sub(self, dur: Duration) -> Self {
        self.checked_sub(dur).expect("overflow when subing duration to instant")
    }
}

impl<const Y: u32> AddAssign<Duration> for Epoch32<Y> {
    fn add_assign(&mut self, other: Duration) {
        *self = *self + other;
    }
}

impl<const Y: u32> SubAssign<Duration> for Epoch32<Y> {
    fn sub_assign(&mut self, other: Duration) {
        *self = *self - other;
    }
}

#[cfg(feature="serde")]
use serde::{Serialize, Deserialize, Serializer, Deserializer};

#[cfg(feature="serde")]
impl<const Y: u32> Serialize for Epoch32<Y> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        serializer.serialize_u64(self.to_unix_epoch())
    }
}

#[cfg(feature="serde")]
impl<'de, const Y: u32> Deserialize<'de> for Epoch32<Y> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: Deserializer<'de>,
    {
        let n = u64::deserialize(deserializer)?;
        Epoch32::<Y>::try_from_unix_epoch(n)
            .ok_or(serde::de::Error::custom("Epoch32 overflow"))
    }
}
