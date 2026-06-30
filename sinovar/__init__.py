# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors: Oier Lauzirika Zarrabeitia (olauzirika@cnb.csic.es)
# *
# * Spanish National Center for Biotechnology (CNB)
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************

import os
import pyworkflow.utils as pwutils
import pwem

from sinovar.constants import *

__version__ = '0.1.0'  # plugin version

class Plugin(pwem.Plugin):
    _url = "https://github.com/scipion-em/scipion-em-sinovar"
    _supportedVersions = VERSIONS # binary version

    @classmethod
    def _defineVariables(cls):
        cls._defineVar(SINOVAR_ENV_ACTIVATION, DEFAULT_ACTIVATION_CMD)

    @classmethod
    def getEnviron(cls):
        """ Setup the environment variables needed to launch my program. """
        environ = pwutils.Environ(os.environ)

        return environ

    @classmethod
    def getDependencies(cls):
        """ Return a list of dependencies. """
        neededProgs = []
        return neededProgs

    @classmethod
    def defineBinaries(cls, env):
        for ver in cls._supportedVersions:
            cls.addSinovarPackage(env, ver, default = (ver == SINOVAR_DEFAULT_VERSION))
                       
    @classmethod
    def addSinovarPackage(cls, env, version, default = False):
        SINOVAR_INSTALLED = f'sinovar_{version}_installed'
        
        condaEnvCmd = cls.getCondaActivationCmd()
        # Environment creation
        SINOVAR_ENV_NAME = f"{SINOVAR_ENV_BASE_NAME}-{version}"
        condaEnvCmd += f' conda create -y -n {SINOVAR_ENV_NAME} python=3.11 && '
        condaEnvCmd += f' conda activate {SINOVAR_ENV_NAME} && '
        # Actual packages installation
        condaEnvCmd += f' pip install git+https://github.com/oierlauzi/sinovar.git jax[cuda12] && '
        condaEnvCmd += f' touch {SINOVAR_INSTALLED}'
        installationCmds = [(condaEnvCmd, SINOVAR_INSTALLED)]

        envPath = os.environ.get('PATH', "")  # keep path since conda likely in there
        installEnvVars = {'PATH': envPath} if envPath else None

        env.addPackage(SINOVAR,
                       version=version,
                       tar='void.tgz',
                       commands=installationCmds,
                       neededProgs=cls.getDependencies(),
                       vars=installEnvVars,
                       default=default)
        
    @classmethod
    def getSinovarEnvActivation(cls):
        return cls.getVar(SINOVAR_ENV_ACTIVATION)

    @classmethod
    def runSinovar(cls, protocol, program, args, cwd=None):
        fullProgram = '%s %s && %s' % (cls.getCondaActivationCmd(),
                                       cls.getSinovarEnvActivation(), program)
        protocol.runJob(fullProgram, args, env=cls.getEnviron(), cwd=cwd,
                        numberOfMpi=1)
                       
