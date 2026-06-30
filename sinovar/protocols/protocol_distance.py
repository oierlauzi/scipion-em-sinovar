# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors: Oier Lauzirika Zarrabeitia (olauzirika@cnb.csic.es)
# * Authors: Mikel Iceta Tena (miceta@cnb.csic.es)
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

import ast
from pyworkflow.constants import BETA
from pyworkflow.protocol.constants import LEVEL_ADVANCED
import pyworkflow.protocol.params as params
from pyworkflow.utils import Message

from pwem.objects import SetOfParticles
from pwem.protocols import EMProtocol

from sinovar import Plugin
from relion.convert import writeSetOfParticles

class SinovarDistance(EMProtocol):
    _label = 'distance'
    _devStatus = BETA
    
    def _defineParams(self, form: params.Form):
        form.addSection(label=Message.LABEL_INPUT)

        form.addHidden(params.USE_GPU, params.BooleanParam, default=True,
                           expertLevel=LEVEL_ADVANCED,
                           label="Use GPU?",
                           help="Set to True if you want to use GPU implementation."
                           )
        
        form.addHidden(params.GPU_LIST, params.StringParam, default='0',
                       expertLevel=LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="GPU may have several cores. Set it to zero"
                            " if you do not know what we are talking about."
                            " First core index is 0, second 1 and so on."
                            " Sinovar can use multiple GPUs - in that case"
                            " set to i.e. *0 1 2*."
                            )

        form.addParam('inputParticles', params.PointerParam,
                      pointerClass = SetOfParticles,
                      pointerCondition = 'hasAlignmentProj', # TODO: Add validation for CTF in the validation
                      important = True,
                      label = 'Input particles',
                      allowsNull = False,
                      help = ''
                      )
        
        form.addParam('diameter', params.FloatParam,
                      label='Diameter (A)')
        form.addParam('resolution', params.FloatParam,
                      label='Resolution (A)')
        
        form.addSection('Compute')
        form.addParam('tileSize', params.IntParam,
                      label='Tile size',
                      default=128 )

    # --------------------------- STEPS functions ------------------------------

    def _insertAllSteps(self):
        self._insertFunctionStep(self.convertInputStep)
        self._insertFunctionStep(self.runSinovarStep)

    def convertInputStep(self):
        writeSetOfParticles(
            self.inputParticles.get(), 
            self._getInputStarFilename()
        )
        
    def runSinovarStep(self):
        program = 'sinovar'

        args = ['distance']
        args += ['-i', self._getInputStarFilename()]
        args += ['-d', self._getDistanceMatrixFilename()]
        args += ['--resolution', self.resolution.get()]
        args += ['--diameter', self.diameter.get()]
        args += ['--block_size', self.tileSize.get()]

        gpus = self.getGpuList()
        if gpus:
            args += ['--device']
            args += list(map('gpu:{:d}'.format, gpus))
            
        Plugin.runSinovar(self, program, args)
        
    # --------------------------- INFO functions ---------------------------------
    # --------------------------- UTILS functions --------------------------------
    def _getInputStarFilename(self) -> str:
        return self._getExtraPath('images.star')

    def _getDistanceMatrixFilename(self) -> str:
        return self._getExtraPath('distances.npy')
