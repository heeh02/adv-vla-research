"""Keep original sensor hook; preserve native Observable.modality metadata."""
from contextlib import contextmanager
from camera_sensor_pair import NativeCameraPairTap,need
class CameraPairTap(NativeCameraPairTap):
    @contextmanager
    def installed(self):
        with super().installed():
            for camera,original in self._old_sensors.items():
                need(hasattr(original,'__modality__'),'actual_sensor_modality_attribute_required')
                wrapped=self.observables[camera+'_image']._sensor
                wrapped.__modality__=original.__modality__
            yield self
