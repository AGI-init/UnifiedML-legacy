# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
"""This file makes it possible to import UnifiedML as a package.

Example:
    In your project, Download the UnifiedML directory:

    $ git clone git@github.com:agi-init/UnifiedML.git

    -------------------------------

    Turn a project file into a UnifiedML-style Run-script that can support all UnifiedML command-line syntax:

    > from UnifiedML.Run import main   Imports UnifiedML
    >
    > if __name__ == '__main__':
    >    main()  # For launching UnifiedML

    Now you can launch your project file with UnifiedML.

    -------------------------------

    Examples:

    Say your file is called MyRunner.py and includes an architecture called MyEyes as a class. You can run:

    $ python MyRunner.py Eyes=MyRunner.MyEyes

    or even define your own recipe MyRecipe.yaml in your local Hyperparams/task/ directory:

    $ python MyRunner.py task=MyRecipe

"""
import sys
import os
import inspect


# Imports UnifiedML paths and the paths of any launching app
def import_paths():
    UnifiedML = os.path.dirname(__file__)
    app = '/'.join(str(inspect.stack()[-1][1]).split('/')[:-1])

    sys.path.extend([UnifiedML, app])  # Imports UnifiedML paths and the paths of the launching app

    if os.path.exists(app + '/Hyperparams'):
        sys.argv.extend(['-cd', app + '/Hyperparams'])  # Adds an app's Hyperparams dir to Hydra's .yaml search path


import_paths()
