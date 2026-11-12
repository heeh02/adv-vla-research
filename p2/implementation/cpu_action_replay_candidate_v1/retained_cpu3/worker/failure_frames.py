"""Bounded code locations only. Never capture locals, source lines or values."""
from pathlib import Path
def frames(error):
    result=[];tb=error.__traceback__
    while tb is not None:
        code=tb.tb_frame.f_code;name=code.co_filename
        name=name.split('/package/',1)[1]if '/package/'in name else Path(name).name
        result.append({'file':name[:240],'function':code.co_name[:100],'line':int(tb.tb_lineno)})
        tb=tb.tb_next
    return result[-12:]
