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
    def test_add_2ad(self):
        from pysimplead import PySAD
        x1 = PySAD(name="x",value=2.0,maxvar=2)
        x2 = PySAD(name="y",value=3.0)
        x3 = x1 + x2
        x3.print()
        self.assertEqual(x3.values[0],5.0)
        self.assertEqual(len(x3.values[1]),2)
        self.assertEqual(x3.values[1][0],1.0)
        self.assertEqual(x3.values[1][1],1.0)
        sys.modules.pop("pysimplead")

    def test_add_ad_and_constant(self):
        from pysimplead import PySAD
        x1 = PySAD(name="x",value=2.0,maxvar=2)
        x3 = x1 + 1
        x3.print()
        self.assertEqual(x3.values[0],3.0)
        self.assertEqual(len(x3.values[1]),2)
        self.assertEqual(x3.values[1][0],1.0)
        self.assertEqual(x3.values[1][1],0.0)
        sys.modules.pop("pysimplead")

    def test_sub_2ad(self):
        from pysimplead import PySAD
        x1 = PySAD(name="x",value=2.0,maxvar=2)
        x2 = PySAD(name="y",value=3.0)
        x3 = x1 - x2
        x3.print()
        self.assertEqual(x3.values[0],-1.0)
        self.assertEqual(len(x3.values[1]),2)
        self.assertEqual(x3.values[1][0],1.0)
        self.assertEqual(x3.values[1][1],-1.0)
        sys.modules.pop("pysimplead")

    def test_sub_ad_and_constant(self):
        from pysimplead import PySAD
        x1 = PySAD(name="x",value=2.0,maxvar=2)
        x3 = x1 - 1
        x3.print()
        self.assertEqual(x3.values[0],1.0)
        self.assertEqual(len(x3.values[1]),2)
        self.assertEqual(x3.values[1][0],1.0)
        self.assertEqual(x3.values[1][1],0.0)
        sys.modules.pop("pysimplead")

if __name__ == '__main__':
    unittest.main()
