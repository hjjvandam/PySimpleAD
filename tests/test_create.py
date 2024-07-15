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

from pysimplead import PySAD
import unittest

class TestAD(unittest.TestCase):
    def test_create_one(self):
        x1 = PySAD(name="x",value=2.0,maxvar=2)
        x2 = PySAD(name="y",value=3.0)
        x3 = PySAD(value=1.5)
        x1.print()
        x2.print()
        x3.print()
        #
        self.assertEqual(x1.names[0],"x")
        self.assertEqual(x1.values[0],2.0)
        self.assertEqual(len(x1.values[1]),2)
        self.assertEqual(x1.values[1][0],1.0)
        self.assertEqual(x1.values[1][1],0.0)
        self.assertEqual(x1.maxvar,2)
        self.assertEqual(x1.numvar,2)
        self.assertEqual(x1.varnum,0)
        #
        self.assertEqual(x2.names[1],"y")
        self.assertEqual(x2.values[0],3.0)
        self.assertEqual(len(x2.values[1]),2)
        self.assertEqual(x2.values[1][0],0.0)
        self.assertEqual(x2.values[1][1],1.0)
        self.assertEqual(x2.maxvar,2)
        self.assertEqual(x2.numvar,2)
        self.assertEqual(x2.varnum,1)
        #
        self.assertEqual(x3.values[0],1.5)
        self.assertEqual(len(x3.values[1]),2)
        self.assertEqual(x3.values[1][0],0.0)
        self.assertEqual(x3.values[1][1],0.0)
        self.assertEqual(x3.maxvar,2)
        self.assertEqual(x3.numvar,2)
        self.assertEqual(x3.varnum,-1)

if __name__ == '__main__':
    unittest.main()
