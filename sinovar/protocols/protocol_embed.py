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

from pwem.objects import SetOfParticles, Particle, SetOfParticlesFlex, ParticleFlex
from pwem.protocols import EMProtocol
from emtable import Table

from sinovar import Plugin
from sinovar.constants import SINOVAR
from .protocol_distance import SinovarDistance

OUTPUT_PARTICLES = 'particles'

class SinovarEmbed(EMProtocol):
    _label = 'embed'
    _devStatus = BETA
    __possible_outputs = {
        OUTPUT_PARTICLES: SetOfParticlesFlex
    }
    
    def _defineParams(self, form: params.Form):
        form.addSection(label=Message.LABEL_INPUT)

        form.addParam('distanceProtocol', params.PointerParam,
                      pointerClass = SinovarDistance,
                      important = True,
                      label = 'Distance protocol',
                      allowsNull = False,
                      )
        
        form.addParam('components', params.IntParam,
                      label='Number of embedding components',
                      default=8 )
        form.addParam('sigmaNn', params.IntParam,
                      label='Number of NN used to estimate sigma',
                      default=16 )
        form.addParam('affinityNn', params.IntParam,
                      label='Number of NN used for the affinity matrix',
                      default=4096 )

    # --------------------------- STEPS functions ------------------------------

    def _insertAllSteps(self):
        self._insertFunctionStep(self.runSinovarStep)
        self._insertFunctionStep(self.createOutputStep)

    def runSinovarStep(self):
        program = 'sinovar'

        args = ['embed']
        args += ['-i', self._getInputStarFilename()]
        args += ['-d', self._getDistanceMatrixFilename()]
        args += ['-o', self._getOutputStarFilename()]
        args += ['--components', self.components.get()]
        args += ['--sigma_k', self.sigmaNn.get()]
        args += ['--affinity_k', self.affinityNn.get()]

        gpus = self.getGpuList()
        if gpus:
            args += ['--device']
            args += list(map('gpu:{:d}'.format, gpus))
            
        Plugin.runSinovar(self, program, args)
        
    def createOutputStep(self):
        inputParticles: SetOfParticles = \
            self._getDistanceProtocol().inputParticles.get()
        outputParticles: SetOfParticlesFlex = SetOfParticlesFlex.create(
            self._getPath(),
            progName=SINOVAR
        )
        outputParticles.copyInfo(inputParticles)
        
        star = Table()
        star.read(
            self._getOutputStarFilename(),
            tableName='particles'
        )
        embeddings = star.getColumnValues('sinovarEmbedding')
        
        particle: Particle
        for particle, embedding in zip(inputParticles, embeddings):
            flexParticle = ParticleFlex(SINOVAR)
            flexParticle.copyInfo(particle)
            flexParticle.setZFlex(ast.literal_eval(embedding))
            outputParticles.append(flexParticle)
        
        self._defineOutputs(**{OUTPUT_PARTICLES: outputParticles})
        self._defineSourceRelation(
            self._getDistanceProtocol().inputParticles, 
            outputParticles
        )

    # -------------------------- INFO functions --------------------------------
    # -------------------------- UTILS functions -------------------------------
    def _getDistanceProtocol(self) -> SinovarDistance:
        return self.distanceProtocol.get()
        
    def _getInputStarFilename(self) -> str:
        return self._getDistanceProtocol()._getInputStarFilename()

    def _getDistanceMatrixFilename(self) -> str:
        return self._getDistanceProtocol()._getDistanceMatrixFilename()

    def _getOutputStarFilename(self) -> str:
        return self._getExtraPath('embedding.star')
