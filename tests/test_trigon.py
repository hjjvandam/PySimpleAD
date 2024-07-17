#   Copyright 2024 Hubertus van Dam 
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#      
#       http://www.apache.org/licenses/LICENSE-2.0
#      
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import unittest
import sys

class TestAD(unittest.TestCase):
    def test_sin(self):
        import math
        from pysimplead import PySAD, sin
        pi  = math.acos(-1.0)
        x0  = PySAD(name="p",value=0.0,maxvar=5)
        x12 = PySAD(name="q",value=pi/2.0)
        x1  = PySAD(name="s",value=pi)
        x32 = PySAD(name="t",value=3.0*pi/2.0)
        x2  = PySAD(name="u",value=2.0*pi)
        y0  = 0.0
        y12 = pi/2.0
        y1  = pi
        y32 = 3.0*pi/2.0
        y2  = 2.0*pi
        a0  = sin(x0)
        a12 = sin(x12)
        a1  = sin(x1)
        a32 = sin(x32)
        a2  = sin(x2)
        b0  = sin(y0)
        b12 = sin(y12)
        b1  = sin(y1)
        b32 = sin(y32)
        b2  = sin(y2)
        self.assertAlmostEqual(a0.values[0] ,b0,  delta=1.0e-10)
        self.assertAlmostEqual(a12.values[0],b12, delta=1.0e-10)
        self.assertAlmostEqual(a1.values[0] ,b1,  delta=1.0e-10)
        self.assertAlmostEqual(a32.values[0],b32, delta=1.0e-10)
        self.assertAlmostEqual(a2.values[0] ,b2,  delta=1.0e-10)
        self.assertAlmostEqual(a0.values[1][0] , 1.0, delta=1.0e-10)
        self.assertAlmostEqual(a12.values[1][1], 0.0, delta=1.0e-10)
        self.assertAlmostEqual(a1.values[1][2] ,-1.0, delta=1.0e-10)
        self.assertAlmostEqual(a32.values[1][3], 0.0, delta=1.0e-10)
        self.assertAlmostEqual(a2.values[1][4] , 1.0, delta=1.0e-10)
        sys.modules.pop("pysimplead")

    def test_cos(self):
        import math
        from pysimplead import PySAD, cos
        pi  = math.acos(-1.0)
        x0  = PySAD(name="p",value=0.0,maxvar=5)
        x12 = PySAD(name="q",value=pi/2.0)
        x1  = PySAD(name="s",value=pi)
        x32 = PySAD(name="t",value=3.0*pi/2.0)
        x2  = PySAD(name="u",value=2.0*pi)
        y0  = 0.0
        y12 = pi/2.0
        y1  = pi
        y32 = 3.0*pi/2.0
        y2  = 2.0*pi
        a0  = cos(x0)
        a12 = cos(x12)
        a1  = cos(x1)
        a32 = cos(x32)
        a2  = cos(x2)
        b0  = cos(y0)
        b12 = cos(y12)
        b1  = cos(y1)
        b32 = cos(y32)
        b2  = cos(y2)
        self.assertAlmostEqual(a0.values[0] ,b0,  delta=1.0e-10)
        self.assertAlmostEqual(a12.values[0],b12, delta=1.0e-10)
        self.assertAlmostEqual(a1.values[0] ,b1,  delta=1.0e-10)
        self.assertAlmostEqual(a32.values[0],b32, delta=1.0e-10)
        self.assertAlmostEqual(a2.values[0] ,b2,  delta=1.0e-10)
        self.assertAlmostEqual(a0.values[1][0] , 0.0, delta=1.0e-10)
        self.assertAlmostEqual(a12.values[1][1],-1.0, delta=1.0e-10)
        self.assertAlmostEqual(a1.values[1][2] , 0.0, delta=1.0e-10)
        self.assertAlmostEqual(a32.values[1][3], 1.0, delta=1.0e-10)
        self.assertAlmostEqual(a2.values[1][4] , 0.0, delta=1.0e-10)
        sys.modules.pop("pysimplead")

    def test_tan(self):
        import math
        from pysimplead import PySAD, tan
        pi  = math.acos(-1.0)
        x0  = PySAD(name="p",value=0.0,maxvar=4)
        x14 = PySAD(name="q",value=pi/4.0)
        x34 = PySAD(name="r",value=3.0*pi/4.0)
        x1  = PySAD(name="s",value=pi)
        y0  = 0.0
        y14 = pi/4.0
        y34 = 3.0*pi/4.0
        y1  = pi
        a0  = tan(x0)
        a14 = tan(x14)
        a34 = tan(x34)
        a1  = tan(x1)
        b0  = tan(y0)
        b14 = tan(y14)
        b34 = tan(y34)
        b1  = tan(y1)
        self.assertAlmostEqual(a0.values[0] ,b0,  delta=1.0e-10)
        self.assertAlmostEqual(a14.values[0],b14, delta=1.0e-10)
        self.assertAlmostEqual(a34.values[0],b34, delta=1.0e-10)
        self.assertAlmostEqual(a1.values[0] ,b1,  delta=1.0e-10)
        self.assertAlmostEqual(a0.values[1][0] , 1.0, delta=1.0e-10)
        self.assertAlmostEqual(a14.values[1][1], 2.0, delta=1.0e-10)
        self.assertAlmostEqual(a34.values[1][2], 2.0, delta=1.0e-10)
        self.assertAlmostEqual(a1.values[1][3] , 1.0, delta=1.0e-10)
        sys.modules.pop("pysimplead")


if __name__ == '__main__':
    unittest.main()
