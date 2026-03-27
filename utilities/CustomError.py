class SyringeDetectionError(Exception):
    def __init__(self, message="Syringe detection error occurred"):
        self.message = message
        super().__init__(self.message)


class ImagecaptureError(Exception):
    def __init__(self, message="Image capture error occurred"):
        self.message = message
        super().__init__(self.message)


class FlatlineError(Exception):
    def __init__(self, message,scan_id):
        self.message = message
        super().__init__(self.message)
        self.scan_id = scan_id


class XcalibrationFailed(Exception):
    def __init__(self, message="X calibration failed\n  not able to find proper raw peaks"):
        self.message = message
        super().__init__(self.message)

class LowIntensityError(Exception):
    def __init__(self,message, scan_id):
        super().__init__(message)
        self.scan_id = scan_id

class HighFluorosenceError(Exception):
    def __init__(self, message="Advance algorithm processing in progress. Could take additional 2 mins. Please Wait"):
        self.message = message
        super().__init__(self.message)

class ShutterFluorosenceError(Exception):
    def __init__(self, message="NarcRanger is unable to provide conclusive results on this sample."):
        self.message = message
        super().__init__(self.message)