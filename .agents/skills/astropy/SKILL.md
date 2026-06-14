---
name: astropy
description: Comprehensive Python library for astronomy and astrophysics. This skill should be used when working with astronomical data including celestial coordinates, physical units, FITS files, cosmological calculations, time systems, tables, world coordinate systems (WCS), and astronomical data analysis. Use when tasks involve coordinate transformations, unit conversions, FITS file manipulation, cosmological distance calculations, time scale conversions, or astronomical data processing.
license: BSD-3-Clause license
metadata:
    skill-author: K-Dense Inc.
---

## Overview
Astropy is the core Python package for astronomy, providing essential functionality for astronomical research and data analysis. Use astropy for coordinate transformations, unit and quantity calculations, FITS file operations, cosmological calculations, precise time handling, tabular data manipulation, and astronomical image processing.

## When to Use This Skill
Use astropy when tasks involve:
- Converting between celestial coordinate systems (ICRS, Galactic, FK5, AltAz, etc.)
- Working with physical units and quantities (converting Jy to mJy, parsecs to km, etc.)
- Reading, writing, or manipulating FITS files (images or tables)
- Cosmological calculations (luminosity distance, lookback time, Hubble parameter)
- Precise time handling with different time scales (UTC, TAI, TT, TDB) and formats (JD, MJD, ISO)
- Table operations (reading catalogs, cross-matching, filtering, joining)
- WCS transformations between pixel and world coordinates
- Astronomical constants and calculations

## Quick Start
```python
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import Planck18

# Units and quantities
distance = 100 * u.pc
distance_km = distance.to(u.km)

# Coordinates
coord = SkyCoord(ra=10.5*u.degree, dec=41.2*u.degree, frame='icrs')
coord_galactic = coord.galactic

# Time
t = Time('2023-01-15 12:30:00')
jd = t.jd  # Julian Date

# FITS files
data = fits.getdata('image.fits')
header = fits.getheader('image.fits')

# Tables
table = Table.read('catalog.fits')

# Cosmology
d_L = Planck18.luminosity_distance(z=1.0)
```

## Key Modules

### 1. Units and Quantities (`astropy.units`)
Handle physical quantities with units, perform unit conversions, and ensure dimensional consistency in calculations.

**Key operations:**
- Create quantities by multiplying values with units
- Convert between units using `.to()` method
- Perform arithmetic with automatic unit handling
- Use equivalencies for domain-specific conversions (spectral, doppler, parallax)
- Work with logarithmic units (magnitudes, decibels)

### 2. Coordinate Systems (`astropy.coordinates`)
Represent celestial positions and transform between different coordinate frames.

**Key operations:**
- Create coordinates with `SkyCoord` in any frame (ICRS, Galactic, FK5, AltAz, etc.)
- Transform between coordinate systems
- Calculate angular separations and position angles
- Match coordinates to catalogs
- Include distance for 3D coordinate operations
- Handle proper motions and radial velocities
- Query named objects from online databases

### 3. Time Handling (`astropy.time`)
Precise time representation and conversion between time scales and formats.

**Key operations:**
- Create Time objects in various formats (ISO, JD, MJD, Unix, etc.)
- Convert between time scales (UTC, TAI, TT, TDB, etc.)
- Perform time arithmetic with TimeDelta
- Calculate sidereal time for observers
- Compute light travel time corrections (barycentric, heliocentric)
- Work with time arrays efficiently
- Handle masked (missing) times

### 4. FITS File Handling (`astropy.io.fits`)
Read, write, and manipulate FITS (Flexible Image Transport System) files.

**Key operations:**
- Open FITS files with context managers
- Access HDUs (Header Data Units) by index or name
- Read and modify headers (keywords, comments, history)
- Work with image data (NumPy arrays)
- Handle table data (binary and ASCII tables)
- Create new FITS files (single or multi-extension)
- Use memory mapping for large files

### 5. Table Operations (`astropy.table`)
Work with tabular data with support for units, metadata, and various file formats.

**Key operations:**
- Create tables from arrays, lists, or dictionaries
- Read/write tables in multiple formats (FITS, CSV, HDF5, VOTable)
- Access and modify columns and rows
- Sort, filter, and index tables
- Perform database-style operations (join, group, aggregate)

### 6. Constants
Physical and astronomical constants with proper units (speed of light, solar mass, Planck constant, etc.).

## Installation
```bash
uv pip install astropy

# With optional dependencies for full functionality
uv pip install astropy[all]
```

## Code Examples

### Converting Coordinates Between Systems
```python
from astropy.coordinates import SkyCoord
import astropy.units as u

# Create coordinate
c = SkyCoord(ra='05h23m34.5s', dec='-69d45m22s', frame='icrs')

# Transform to galactic
c_gal = c.galactic
print(f"l={c_gal.l.deg}, b={c_gal.b.deg}")

# Transform to alt-az (requires time and location)
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz

observing_time = Time('2023-06-15 23:00:00')
observing_location = EarthLocation(lat=40*u.deg, lon=-120*u.deg)
aa_frame = AltAz(obstime=observing_time, location=observing_location)
c_altaz = c.transform_to(aa_frame)
print(f"Alt={c_altaz.alt.deg}, Az={c_altaz.az.deg}")
```

### Cosmological Distance Calculations
```python
from astropy.cosmology import Planck18
import astropy.units as u

# Calculate distances at z=1.5
z = 1.5
d_L = Planck18.luminosity_distance(z)
d_A = Planck18.angular_diameter_distance(z)

print(f"Luminosity distance: {d_L}")
print(f"Angular diameter distance: {d_A}")

# Age of universe at that redshift
age = Planck18.age(z)
print(f"Age at z={z}: {age.to(u.Gyr)}")

# Lookback time
t_lookback = Planck18.lookback_time(z)
print(f"Lookback time: {t_lookback.to(u.Gyr)}")
```

## Best Practices
1. **Always use units**: Attach units to quantities to avoid errors and ensure dimensional consistency
2. **Use context managers for FITS files**: Ensures proper file closing
3. **Prefer arrays over loops**: Process multiple coordinates/times as arrays for better performance
4. **Check coordinate frames**: Verify the frame before transformations
5. **Specify time scales**: Be explicit about time scales (UTC, TT, TDB) for precise timing
6. **Cache frequently used values**: Expensive calculations (e.g., cosmological distances) can be cached

## Documentation and Resources
- Official Astropy Documentation: https://docs.astropy.org/en/stable/
- Tutorials: https://learn.astropy.org/
- GitHub: https://github.com/astropy/astropy
