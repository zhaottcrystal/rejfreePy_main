 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 15:10:43 2017

@author: crystal
"""

class CollisionContext:

    def __init__(self, prng, velocity):
        """ This class has one attributes:
            velocity: is a vector of velocities. It should be the same length 
                      as the number of parameters.
        """
        self.velocity = velocity
        self.prng = prng