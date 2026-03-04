import datetime
from dataclasses import dataclass
from collections import defaultdict

import pandas as pd
from pyproj import Geod
import shapely.geometry as geom
import shapely.ops as ops

class GTFSLoader:
    def __init__(self, folder: str, date_str: str = None):
        # Read GTFS files
        self.stops = pd.read_csv(f"{folder}/stops.txt", dtype=str)
        self.stop_times = pd.read_csv(f"{folder}/stop_times.txt", dtype=str)
        self.trips = pd.read_csv(f"{folder}/trips.txt", dtype=str)
        self.routes = pd.read_csv(f"{folder}/routes.txt", dtype=str)
        self.shapes = pd.read_csv(f"{folder}/shapes.txt", dtype=str)

        # If a date filter is provided, compute active services and filter trips/stop_times
        if date_str is not None:
            active = self._get_active_service_ids(folder, date_str)
            if len(active) == 0:
                print(f"Warning: no active services found for {date_str}; resulting GTFS will be empty.")
                # keep empty trips
                self.trips = self.trips.iloc[0:0]
            else:
                # trips.csv should have service_id column
                if "service_id" in self.trips.columns:
                    self.trips = self.trips[self.trips.service_id.isin(active)].copy()
                else:
                    print("Warning: trips.txt has no 'service_id' column; cannot filter by date.")

                valid_trip_ids = set(self.trips.trip_id.unique())
                # filter stop_times to only those trips
                self.stop_times = self.stop_times[self.stop_times.trip_id.isin(valid_trip_ids)].copy()

        # Ensure numeric types and compute seconds
        if not self.stop_times.empty:
            self.stop_times["stop_sequence"] = self.stop_times["stop_sequence"].astype(int)

            # Some GTFS files may have times with >24:00:00; handle safely by parsing hours
            self.stop_times["arrival_secs"] = self.stop_times["arrival_time"].apply(self._hhmmss)
            self.stop_times["departure_secs"] = self.stop_times["departure_time"].apply(self._hhmmss)
        else:
            # Create the columns to avoid KeyErrors later
            self.stop_times["stop_sequence"] = pd.Series(dtype=int)
            self.stop_times["arrival_secs"] = pd.Series(dtype=float)
            self.stop_times["departure_secs"] = pd.Series(dtype=float)

        self.shapes["shape_pt_lat"] = self.shapes["shape_pt_lat"].astype(float)
        self.shapes["shape_pt_lon"] = self.shapes["shape_pt_lon"].astype(float)
        self.shapes["shape_pt_sequence"] = self.shapes["shape_pt_sequence"].astype(int)

        self.shape_polylines = self._build_shape_polylines()

    @staticmethod
    def _hhmmss(ts):
        # Robust hh:mm:ss parser that accepts hours >= 0 (including >23)
        try:
            parts = ts.split(":")
            h = int(parts[0])
            m = int(parts[1])
            s = int(parts[2])
            return h * 3600 + m * 60 + s
        except Exception:
            return 0

    def _get_active_service_ids(self, folder: str, date_str: str):
        """Return set of service_ids active on date_str (YYYYMMDD) using calendar.txt and calendar_dates.txt."""
        target_date = datetime.datetime.strptime(date_str, "%Y%m%d").date()
        services = set()

        cal_path = f"{folder}/calendar.txt"
        cal_dates_path = f"{folder}/calendar_dates.txt"

        try:
            cal = pd.read_csv(cal_path, dtype=str)
            cal["start_date"] = pd.to_datetime(cal["start_date"], format="%Y%m%d").dt.date
            cal["end_date"] = pd.to_datetime(cal["end_date"], format="%Y%m%d").dt.date

            weekday_name = target_date.strftime("%A").lower()
            for _, row in cal.iterrows():
                if row["start_date"] <= target_date <= row["end_date"]:
                    if str(row.get(weekday_name, "0")) == "1":
                        services.add(row["service_id"])
        except FileNotFoundError:
            pass
        except Exception:
            # If calendar.txt malformed, continue with calendar_dates only
            pass

        try:
            cal_dates = pd.read_csv(cal_dates_path, dtype=str)
            # keep only rows for this date
            rows = cal_dates[cal_dates["date"] == date_str]
            for _, r in rows.iterrows():
                sid = r["service_id"]
                ex = r.get("exception_type", "1")
                if ex == "1":
                    services.add(sid)
                elif ex == "2" and sid in services:
                    services.discard(sid)
        except FileNotFoundError:
            pass
        except Exception:
            pass

        return services

    def _build_shape_polylines(self):
        polylines = {}
        # create a local geod for metric distance computations
        geod = Geod(ellps="WGS84")
        for sid, grp in self.shapes.groupby("shape_id"):
            grp = grp.sort_values("shape_pt_sequence")
            coords_latlon = list(zip(grp.shape_pt_lat.astype(float), grp.shape_pt_lon.astype(float)))
            # shapely expects (x,y) == (lon,lat)
            line = geom.LineString([(lon, lat) for lat, lon in coords_latlon])

            # compute total metric length (meters) along the polyline using geod
            seg_meters = []
            seg_units = []
            cum_units = [0.0]
            cum_m = [0.0]
            total_m = 0.0
            total_units = 0.0
            for i in range(len(coords_latlon) - 1):
                lat1, lon1 = coords_latlon[i]
                lat2, lon2 = coords_latlon[i + 1]
                # Geod.inv expects (lon, lat)
                _, _, d_m = geod.inv(lon1, lat1, lon2, lat2)
                seg_m = abs(d_m)
                seg_meters.append(seg_m)

                # unit length for the segment in shapely coordinate units (lon/lat degrees)
                seg_line = geom.LineString([(lon1, lat1), (lon2, lat2)])
                seg_u = seg_line.length
                seg_units.append(seg_u)

                total_m += seg_m
                total_units += seg_u
                cum_units.append(total_units)
                cum_m.append(total_m)

            polylines[sid] = {
                "coords": coords_latlon,    # list of (lat, lon)
                "line": line,               # shapely LineString (lon,lat)
                "total_m": total_m,         # total metric length in meters
                "line_length_units": total_units,  # length in coordinate units (degrees)
                "seg_units": seg_units,
                "seg_meters": seg_meters,
                "cum_units": cum_units,
                "cum_m": cum_m,
            }

        return polylines

    def stop_sequence(self, trip_id: str):
        df = self.stop_times[self.stop_times.trip_id == trip_id].sort_values("stop_sequence")
        seq = []
        for _, r in df.iterrows():
            s = self.stops[self.stops.stop_id == r.stop_id].iloc[0]
            seq.append({
                "stop_id": r.stop_id,
                "arrival": r.arrival_secs,
                "lat": float(s.stop_lat),
                "lon": float(s.stop_lon),
            })
        return seq

    def shape_for_trip(self, trip_id):
        row = self.trips[self.trips.trip_id == trip_id]
        return None if row.empty else row.iloc[0].shape_id