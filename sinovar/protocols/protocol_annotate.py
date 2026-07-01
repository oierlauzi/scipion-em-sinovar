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

from pwem.objects import (
    Integer, 
    SetOfParticles, Particle, 
    SetOfParticlesFlex, ParticleFlex
)
from pwem.protocols import EMProtocol

from sinovar import Plugin

from emtable import Table
from relion.convert import writeSetOfParticles

OUTPUT_PARTICLES = 'particles'

def _writeSinovarEmbedding(particle: ParticleFlex, row: dict):
    row['sinovarEmbedding'] = '[' + str(particle._zFlex.get()) + ']'

class SinovarAnnotate(EMProtocol):
    _label = 'annotate'
    _devStatus = BETA
    _possible_outputs = {
        OUTPUT_PARTICLES: SetOfParticles
    }
    
    def _defineParams(self, form: params.Form):
        form.addSection(label=Message.LABEL_INPUT)

        form.addParam('inputParticles', params.PointerParam,
                      pointerClass = SetOfParticlesFlex,
                      important = True,
                      label = 'Flex particles',
                      allowsNull = False,
                      )
        
    # --------------------------- STEPS functions ------------------------------

    def _insertAllSteps(self):
        self._insertFunctionStep(self.convertInputStep)
        self._insertFunctionStep(self.runSinovarStep)
        self._insertFunctionStep(self.createOutputStep)

    def convertInputStep(self):
        writeSetOfParticles(
            self.inputParticles.get(), 
            self._getInputStarFilename(),
            postprocessImageRow=_writeSinovarEmbedding
        )
    
    def runSinovarStep(self):
        program = 'sinovar'

        args = ['annotate']
        args += ['-i', self._getInputStarFilename()]
        args += ['-o', self._getOutputStarFilename()]
            
        Plugin.runSinovar(self, program, args)
        
    def createOutputStep(self):
        inputParticles: SetOfParticlesFlex = \
            self.inputParticles.get()
        outputParticles: SetOfParticles = SetOfParticles.create(self._getPath())
        outputParticles.copyInfo(inputParticles)
        
        star = Table()
        star.read(
            self._getOutputStarFilename(),
            tableName='particles'
        )
        classIds = star.getColumnValues('sinovarClassId')
        
        particle: ParticleFlex
        for particle, classId in zip(inputParticles, classIds):
            outputParticle = Particle()
            outputParticle.copy(particle)
            outputParticle.setClassId(classId+1)
            outputParticles.append(outputParticle)
        
        self._defineOutputs(**{OUTPUT_PARTICLES: outputParticles})
        self._defineSourceRelation(
            self.inputParticles, 
            outputParticles
        )
    
    # -------------------------- INFO functions --------------------------------
    # -------------------------- UTILS functions -------------------------------
    def _getInputStarFilename(self) -> str:
        return self._getExtraPath('particles.star')

    def _getOutputStarFilename(self) -> str:
        return self._getExtraPath('annotated.star')
