
from pathlib import Path
import appdirs

from flame.util import utils


def config(path=None) -> None:
    """Configures model repository.

    Loads config.yaml and writes a correct model repository path
    with the path provided by the user or a default from appdirs
    if the path is not provided.
    """
    if path is None:  # set default
        default_models_path = Path(appdirs.user_data_dir('flame_models'))

        print(f'Setting model repository (default) to {default_models_path}'
              'Would you like to continue?(y/n)')
        
        userinput = input()
        
        if userinput.lower() not in ['yes', 'no', 'y', 'n']:
                print('Please write "yes", "no", "y" or "n"')
                return

        elif userinput.lower() in ['yes', 'y']:
            if default_models_path.exists():
                print(f'{default_models_path} already exists. '
                      'Would you like to set is as model repository anyway?(y/n)')

                userinput = input()

                if userinput.lower() not in ['yes', 'no', 'y', 'n']:
                    print('Please write "yes", "no", "y" or "n"')
                    return

                elif userinput.lower() in ['yes', 'y']:
                    utils.set_model_repository(default_models_path)

                else:
                    print('aborting...')
                    return

            else:  # models_path doesn't exists
                default_models_path.mkdir(parents=True)
                utils.set_model_repository(default_models_path)

            print(f'model repository set to {default_models_path}')

        elif userinput.lower() in ['no', 'n']:
            print('aborting...')
            return

    else:  # path input by user
        in_path = Path(path).expanduser()
        current_models_path = Path(utils.model_repository_path())

        if in_path == current_models_path:
            print(f'{in_path} already is model repository path')
            return

        elif not in_path.exists():
            print(f"{in_path} doesn't exists. Would you like to create it?(y/n)")

            userinput = input()

            if userinput.lower() not in ['yes', 'no', 'y', 'n']:
                print('Please write "yes", "no", "y" or "n"')
                return

            elif userinput.lower() in ['yes', 'y']:
                in_path.mkdir(parents=True)
                utils.set_model_repository(in_path)

            else:
                print('aborting...')
                return

        else:  # in_path exists
            utils.set_model_repository(in_path)

        print(f'model repository set to {in_path}')
