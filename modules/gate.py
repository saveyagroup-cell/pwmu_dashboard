"""
Gate 1: ANPR & Vehicle Counter combined on a single camera feed.
Internally reuses the existing VehicleProcessor and PlateProcessor —
no logic duplicated, just composed together so one physical gate camera
gives both the IN/OUT count AND the plate reading in one video.
"""
from modules.vehicle import VehicleProcessor
from modules.plate import PlateProcessor


class GateProcessor:
    def __init__(self):
        self.vehicle = VehicleProcessor()
        self.plate = PlateProcessor()

    def process(self, frame):
        frame = self.vehicle.process(frame)
        vehicle_context = self.vehicle.get_recent_crossing()
        frame = self.plate.process(frame, vehicle_context=vehicle_context)
        return frame

    def reset(self):
        self.vehicle.reset()

    def manual_entry(self, plate_text, vehicle_type="—", direction="—"):
        self.plate.manual_entry(plate_text, vehicle_type, direction)

    def get_counts(self):
        c = self.vehicle.get_counts()
        c["recent_plates"] = self.plate.get_recent()
        return c

    def get_recent(self):
        return self.plate.get_recent()
