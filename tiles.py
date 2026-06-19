#!/usr/bin/env python3

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
import csv


def degrees_to_hms(degrees):
    """
    Convert RA in degrees to HMS format string using astropy.
    
    Parameters:
    -----------
    degrees : float
        Right Ascension in degrees (0-360)
    
    Returns:
    --------
    hms_str : str
        RA in format HH:MM:SS.S
    """
    coord = SkyCoord(ra=degrees*u.deg, dec=0*u.deg, frame='icrs')
    ra_hms = coord.ra.to_string(unit=u.hour, sep=':', precision=1, pad=True)
    return ra_hms

def degrees_to_dms(degrees):
    """
    Convert Dec in degrees to DMS format string using astropy.
    
    Parameters:
    -----------
    degrees : float
        Declination in degrees (-90 to +90)
    
    Returns:
    --------
    dms_str : str
        Dec in format +DD:MM:SS.S or -DD:MM:SS.S
    """
    coord = SkyCoord(ra=0*u.deg, dec=degrees*u.deg, frame='icrs')
    dec_dms = coord.dec.to_string(unit=u.deg, sep=':', precision=1, 
                                   alwayssign=True, pad=True)
    return dec_dms

def write_target_scheduler_csv(rectangle_data, filename='survey_targets.csv'):
    """
    Write rectangle data to CSV file in NINA Target Scheduler format.
    
    Parameters:
    -----------
    rectangle_data : list of tuples
        List of (ra, dec, region, constellation, name) tuples
    filename : str
        Output CSV filename
    """
    fieldnames = ['Name', 'Ra', 'Dec', 'Rotation', 'ROI']
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        i = 0
        while i < len(rectangle_data):
            ra, dec, region, constellation, name = rectangle_data[i]
            
            ra_hms = degrees_to_hms(ra)
            dec_dms = degrees_to_dms(dec)
            
            row = {
                'Name': name,
                'Ra': ra_hms,
                'Dec': dec_dms,
                'Rotation': '0',
                'ROI': '100'
            }
            writer.writerow(row)
            i += 1
    
    print("\nWrote {} targets to {}".format(len(rectangle_data), filename))

def get_constellations(ra_values, dec_values):
    """
    Determine constellations for arrays of coordinates.
    """
    coord = SkyCoord(ra=np.asarray(ra_values)*u.deg,
                     dec=np.asarray(dec_values)*u.deg,
                     frame='icrs')
    try:
        return np.asarray(coord.get_constellation(short_name=True))
    except Exception:
        return np.array(['Unknown'] * len(ra_values))

def create_rectangular_tiling(fov_width=15.0, fov_height=10.0, overlap=2.0,
                              polar_dec_limit=75.0):
    """
    Create the historical all-sky tiling used by good_tiles_NMWTTUQ1.

    The four-field polar caps and transition rings avoid poor coverage near the
    poles. Declinations are inverted at the end to preserve the established
    field ordering and place the less regular transition in the south.
    """
    if overlap >= fov_width or overlap >= fov_height:
        raise ValueError("overlap must be smaller than both FoV dimensions")
    centers = []
    half_height = fov_height / 2.0
    dec_step = fov_height - overlap

    south_pole_dec = -90.0 + half_height
    for ra in [0.0, 90.0, 180.0, 270.0]:
        centers.append((ra, south_pole_dec, 'south_polar'))

    south_polar_top = south_pole_dec + half_height
    equatorial_south_dec = -polar_dec_limit + half_height
    equatorial_south_bottom = equatorial_south_dec - half_height
    gap_south = equatorial_south_bottom - south_polar_top
    if gap_south > 0.5:
        trans_dec = (south_polar_top + equatorial_south_bottom) / 2.0
        cos_dec = np.cos(np.radians(trans_dec))
        n_ra = max(1, int(np.ceil(360.0 / ((fov_width - overlap) / cos_dec))))
        for ra in np.arange(n_ra, dtype=float) * (360.0 / n_ra):
            centers.append((float(ra), trans_dec, 'south_transition'))

    dec = equatorial_south_dec
    while dec <= polar_dec_limit - half_height:
        cos_dec = np.cos(np.radians(dec))
        n_ra = max(1, int(np.ceil(360.0 / ((fov_width - overlap) / cos_dec))))
        ra_step = 360.0 / n_ra
        for ra in np.arange(n_ra, dtype=float) * ra_step:
            centers.append((float(ra), float(dec), 'equatorial'))
        dec += dec_step

    last_equatorial_dec = dec - dec_step
    equatorial_north_top = last_equatorial_dec + half_height
    north_pole_dec = 90.0 - half_height
    north_polar_bottom = north_pole_dec - half_height
    gap_north = north_polar_bottom - equatorial_north_top
    if gap_north > 0.5:
        trans_dec = (equatorial_north_top + north_polar_bottom) / 2.0
        cos_dec = np.cos(np.radians(trans_dec))
        n_ra = max(1, int(np.ceil(360.0 / ((fov_width - overlap) / cos_dec))))
        for ra in np.arange(n_ra, dtype=float) * (360.0 / n_ra):
            centers.append((float(ra), trans_dec, 'north_transition'))

    for ra in [0.0, 90.0, 180.0, 270.0]:
        centers.append((ra, north_pole_dec, 'north_polar'))

    region_map = {
        'south_polar': 'north_polar',
        'north_polar': 'south_polar',
        'south_transition': 'north_transition',
        'north_transition': 'south_transition',
        'equatorial': 'equatorial',
    }
    return [(ra, -dec, region_map[region]) for ra, dec, region in centers]


def filter_tiling(centers, min_dec=None, min_gal_lat=None, max_gal_lat=None,
                  min_av=None, sfd_model=None):
    """Apply optional center-coordinate filters to an existing tiling.

    The slow SFD map is imported and evaluated only when ``min_av`` is set.
    """
    if not centers:
        return []

    ra = np.asarray([center[0] for center in centers])
    dec = np.asarray([center[1] for center in centers])
    keep = np.ones(len(centers), dtype=bool)
    if min_dec is not None:
        keep &= dec > min_dec

    need_galactic = (min_gal_lat is not None or max_gal_lat is not None or
                     min_av is not None)
    if need_galactic:
        galactic = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs').galactic
        gal_l = galactic.l.deg
        gal_b = galactic.b.deg
        if min_gal_lat is not None:
            keep &= gal_b >= min_gal_lat
        if max_gal_lat is not None:
            keep &= gal_b <= max_gal_lat

        if min_av is not None:
            if sfd_model is None:
                try:
                    import mwdust
                    sfd_model = mwdust.SFD(filter='Landolt V')
                except Exception as exc:
                    raise RuntimeError(
                        "Could not initialize mwdust.SFD('Landolt V')."
                    ) from exc
            av = np.asarray(sfd_model(gal_l, gal_b, np.ones(len(centers))))
            keep &= av > min_av

    return [center for center, selected in zip(centers, keep) if selected]


def split_by_galactic_latitude(rectangle_data, max_gal_lat=10.0,
                               max_bulge_gal_lat=20.0,
                               max_bulge_gal_long=30.0):
    """Split fields using the disk/bulge cuts from split_in_gal_lat.sh."""
    if not rectangle_data:
        return [], []

    coords = SkyCoord(
        ra=np.asarray([row[0] for row in rectangle_data])*u.deg,
        dec=np.asarray([row[1] for row in rectangle_data])*u.deg,
        frame='icrs',
    ).galactic
    gal_l = coords.l.deg
    gal_b = coords.b.deg
    disk = (-max_gal_lat < gal_b) & (gal_b < max_gal_lat)
    bulge = ((-max_bulge_gal_lat < gal_b) &
             (gal_b < max_bulge_gal_lat) &
             ((gal_l > 360.0 - max_bulge_gal_long) |
              (gal_l < max_bulge_gal_long)))
    low_mask = disk | bulge
    low = [row for row, selected in zip(rectangle_data, low_mask) if selected]
    high = [row for row, selected in zip(rectangle_data, low_mask) if not selected]
    return low, high

def plot_allsky_rectangles(centers, fov_width=15.0, fov_height=10.0):
    """
    Create an all-sky plot with rectangles marked.
    """
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111, projection='mollweide')
    
    # Color code by region
    colors = {
        'equatorial': 'blue', 'north_polar': 'red', 'south_polar': 'green',
        'north_transition': 'orange', 'south_transition': 'cyan',
    }
    
    for ra, dec, region in centers:
        lon = np.radians(ra)
        if lon > np.pi:
            lon -= 2 * np.pi
        # Negate longitude to reverse RA direction (astronomical convention)
        lon = -lon
        lat = np.radians(dec)
        
        half_width = np.radians(fov_width / 2.0)
        half_height = np.radians(fov_height / 2.0)
        
        cos_dec = np.cos(lat)
        if cos_dec > 1e-6:
            ra_width = half_width / cos_dec
        else:
            ra_width = np.pi
        
        n_points = 20
        ra_edge = np.linspace(lon - ra_width, lon + ra_width, n_points)
        ra_edge = np.where(ra_edge > np.pi, ra_edge - 2*np.pi, ra_edge)
        ra_edge = np.where(ra_edge < -np.pi, ra_edge + 2*np.pi, ra_edge)
        
        top_lat = np.clip(lat + half_height, -np.pi/2, np.pi/2)
        bottom_lat = np.clip(lat - half_height, -np.pi/2, np.pi/2)
        
        color = colors.get(region, 'tab:blue')
        ax.plot(ra_edge, np.full_like(ra_edge, top_lat), '-', 
                color=color, linewidth=0.5, alpha=0.6)
        ax.plot(ra_edge, np.full_like(ra_edge, bottom_lat), '-', 
                color=color, linewidth=0.5, alpha=0.6)
        
        if cos_dec > 1e-6:
            dec_edge = np.linspace(bottom_lat, top_lat, n_points)
            ax.plot(np.full_like(dec_edge, lon - ra_width), dec_edge, '-', 
                    color=color, linewidth=0.5, alpha=0.6)
            ax.plot(np.full_like(dec_edge, lon + ra_width), dec_edge, '-', 
                    color=color, linewidth=0.5, alpha=0.6)
        
        ax.plot(lon, lat, '.', color=color, markersize=2)
    
    ax.set_xlabel('RA (degrees)', fontsize=12)
    ax.set_ylabel('Dec (degrees)', fontsize=12)
    ax.set_title(f'All-Sky Tiling: {len(centers)} fields\n'
                 f'FoV = {fov_width:g}x{fov_height:g} deg, '
                 f'landscape orientation',
                 fontsize=12)
    
    ax.set_xticks(np.radians([-150, -120, -90, -60, -30, 0, 
                               30, 60, 90, 120, 150]))
    ax.set_xticklabels(['150', '120', '90', '60', '30', '0',
                        '330', '300', '270', '240', '210'])
    
    plt.tight_layout()
    return fig

def radec_to_vec(ra_deg, dec_deg):
    theta = np.radians(90.0 - dec_deg)
    phi = np.radians(ra_deg % 360.0)
    return hp.ang2vec(theta, phi)

def rectangle_vertices(ra_deg, dec_deg, fov_width, fov_height):
    half_width = fov_width / 2.0
    half_height = fov_height / 2.0
    cos_dec = np.cos(np.radians(dec_deg))
    if cos_dec < 1e-6:
        return None

    ra_offset = min(half_width / cos_dec, 179.999)
    ra_left = (ra_deg - ra_offset) % 360.0
    ra_right = (ra_deg + ra_offset) % 360.0
    dec_top = np.clip(dec_deg + half_height, -90.0, 90.0)
    dec_bottom = np.clip(dec_deg - half_height, -90.0, 90.0)

    return np.array([
        radec_to_vec(ra_left, dec_top),
        radec_to_vec(ra_right, dec_top),
        radec_to_vec(ra_right, dec_bottom),
        radec_to_vec(ra_left, dec_bottom),
    ])

def create_coverage_map(centers, fov_width=15.0, fov_height=10.0, nside=64):
    """
    Create a HEALPix map showing coverage count.
    """
    npix = hp.nside2npix(nside)
    coverage_map = np.zeros(npix, dtype=np.uint16)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    pixel_dec = 90.0 - np.degrees(theta)
    pixel_ra = np.degrees(phi)

    for rect_ra, rect_dec, _ in centers:
        delta_ra = pixel_ra - rect_ra
        delta_ra = np.where(delta_ra > 180.0, delta_ra - 360.0, delta_ra)
        delta_ra = np.where(delta_ra < -180.0, delta_ra + 360.0, delta_ra)
        effective_delta_ra = delta_ra * np.cos(np.radians(rect_dec))
        delta_dec = pixel_dec - rect_dec
        within = ((np.abs(effective_delta_ra) <= fov_width / 2.0) &
                  (np.abs(delta_dec) <= fov_height / 2.0))
        coverage_map[within] += 1

    return coverage_map

# Main execution
if __name__ == "__main__":
    FOV_WIDTH = 15.0           # degrees (RA, long axis East-West)
    FOV_HEIGHT = 10.0          # degrees (Dec, short axis North-South)
    OVERLAP = 2.0              # degrees
    NSIDE = 64                 # ~1 degree pixels
    POLAR_DEC_LIMIT = 75.0     # tiling geometry, not an output constraint
    MIN_DEC = None             # optional strict lower field-center limit
    MIN_GAL_LAT = None         # optional inclusive Galactic latitude limit
    MAX_GAL_LAT = None         # optional inclusive Galactic latitude limit
    MIN_AV = None              # optional strict SFD Landolt-V absorption limit
    SPLIT_BY_GALACTIC_LAT = True
    MAX_SPLIT_GAL_LAT = 10.0
    MAX_BULGE_GAL_LAT = 20.0
    MAX_BULGE_GAL_LONG = 30.0

    print(f"HEALPix NSIDE = {NSIDE}")
    print(f"Pixel size = {hp.nside2resol(NSIDE, arcmin=True):.2f} arcmin")
    print(f"Total pixels = {hp.nside2npix(NSIDE)}")
    print(f"FoV = {FOV_WIDTH:.1f} x {FOV_HEIGHT:.1f} degrees")
    print(f"Requested overlap = {OVERLAP:.1f} degrees")
    print(f"Declination filter: {MIN_DEC}")
    print(f"Galactic latitude filter: {MIN_GAL_LAT} to {MAX_GAL_LAT}")
    print(f"Dust filter: {MIN_AV}")
    print()

    print("Creating all-sky rectangular tiling...")
    centers = create_rectangular_tiling(
        FOV_WIDTH, FOV_HEIGHT, OVERLAP, POLAR_DEC_LIMIT
    )
    centers = filter_tiling(
        centers, min_dec=MIN_DEC, min_gal_lat=MIN_GAL_LAT,
        max_gal_lat=MAX_GAL_LAT, min_av=MIN_AV,
    )

    print(f"\nNumber of fields: {len(centers)}")
    if not centers:
        raise RuntimeError("No fields matched the requested Dec and Galactic latitude limits.")

    ra_values = np.array([c[0] for c in centers], dtype=float)
    dec_values = np.array([c[1] for c in centers], dtype=float)
    print(f"Field center Dec range: {dec_values.min():.2f} to {dec_values.max():.2f} deg")

    print("\nAssigning names to fields...")
    constellation_counts = {}
    rectangle_data = []

    constellations = get_constellations(ra_values, dec_values)
    for (ra, dec, region_name), constellation in zip(centers, constellations):
        if constellation not in constellation_counts:
            constellation_counts[constellation] = 0
        constellation_counts[constellation] += 1

        name = f"{constellation}-{constellation_counts[constellation]:02d}"
        rectangle_data.append((float(ra), float(dec), region_name, constellation, name))

    print(f"\nFirst 20 field centers (RA, Dec in degrees, Constellation, Name):")
    print("=" * 80)

    for i, (ra, dec, region, constellation, name) in enumerate(rectangle_data[:20]):
        print(f"{i+1:4d}: RA = {ra:8.3f}, Dec = {dec:7.3f}, "
              f"Const = {constellation:>3s}, Name = {name}")
    if len(rectangle_data) > 20:
        print(f"... ({len(rectangle_data) - 20} more fields omitted)")

    write_target_scheduler_csv(rectangle_data, 'survey_targets.csv')

    if SPLIT_BY_GALACTIC_LAT:
        low_gal_lat, high_gal_lat = split_by_galactic_latitude(
            rectangle_data,
            max_gal_lat=MAX_SPLIT_GAL_LAT,
            max_bulge_gal_lat=MAX_BULGE_GAL_LAT,
            max_bulge_gal_long=MAX_BULGE_GAL_LONG,
        )
        write_target_scheduler_csv(
            low_gal_lat, 'survey_targets_low_gal_lat.csv'
        )
        write_target_scheduler_csv(
            high_gal_lat, 'survey_targets_high_gal_lat.csv'
        )

    print("\nCreating all-sky visualization...")
    fig1 = plot_allsky_rectangles(centers, FOV_WIDTH, FOV_HEIGHT)
    plt.savefig('allsky_tiling.png', dpi=150, bbox_inches='tight')

    print("Creating coverage map...")
    coverage = create_coverage_map(centers, FOV_WIDTH, FOV_HEIGHT, NSIDE)

    covered_pixels = coverage[coverage > 0]
    print(f"\nCoverage statistics:")
    if covered_pixels.size > 0:
        print(f"  Minimum coverage (covered pixels): {np.min(covered_pixels):.0f}x")
        print(f"  Maximum coverage (covered pixels): {np.max(covered_pixels):.0f}x")
        print(f"  Mean coverage (covered pixels): {np.mean(covered_pixels):.2f}x")
    print(f"  Covered pixels: {covered_pixels.size}")
    print(f"  Uncovered pixels: {np.sum(coverage == 0)}")

    fig2 = plt.figure(figsize=(12, 6))
    hp.mollview(coverage, title='Coverage Count per Pixel (All-Sky Fields)',
                cmap='YlOrRd', flip='astro', hold=True, fig=fig2.number)
    plt.savefig('coverage_map.png', dpi=150, bbox_inches='tight')

    if 'agg' not in plt.get_backend().lower():
        plt.show()

    print("\nDone! Files created:")
    print("  - survey_targets.csv (Target Scheduler import file)")
    if SPLIT_BY_GALACTIC_LAT:
        print("  - survey_targets_low_gal_lat.csv")
        print("  - survey_targets_high_gal_lat.csv")
    print("  - allsky_tiling.png (visualization)")
    print("  - coverage_map.png (coverage map)")
