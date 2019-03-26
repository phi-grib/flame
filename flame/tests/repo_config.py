#import os
#MODEL_REPOSITORY = f"{os.environ.get('HOME')}/testmodels"

import os.path

home_folder = os.path.expanduser('~')
MODEL_REPOSITORY = os.path.join(home_folder,'testmodels')
