import threading 
import traceback
import sys
import os
import tempfile

class FlThread (threading.Thread):
  def __init__ (self, *args, **kwargs):
    self.inner_name = kwargs['name'] 
    super().__init__(*args, **kwargs)
  
  def run (self, *args, **kwargs):
    try:
      super().run (*args, **kwargs)
    except:
      # ceate a file in temp with the exception error inside
      tmp = os.path.join(tempfile.gettempdir(),self.inner_name)
      with open (tmp,'w') as f:
        f.write(traceback.format_exc())
      sys.excepthook(*sys.exc_info())