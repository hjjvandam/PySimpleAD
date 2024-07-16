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
    def test_lt(self):
        from pysimplead import PySAD
        x1 = PySAD(name="x",value=2.0,maxvar=2)
        x2 = PySAD(name="y",value=4.0)
        x3 = 3.0
        bool_x1_lt_x1 = (x1 < x1)
        bool_x1_lt_x2 = (x1 < x2)
        bool_x1_lt_x3 = (x1 < x3)
        bool_x2_lt_x1 = (x2 < x1)
        bool_x2_lt_x3 = (x2 < x3)
        self.assertEqual(bool_x1_lt_x1,False)
        self.assertEqual(bool_x1_lt_x2,True)
        self.assertEqual(bool_x1_lt_x3,True)
        self.assertEqual(bool_x2_lt_x1,False)
        self.assertEqual(bool_x2_lt_x3,False)
        sys.modules.pop("pysimplead")

    def test_le(self):
        from pysimplead import PySAD
        x1 = PySAD(name="x",value=2.0,maxvar=2)
        x2 = PySAD(name="y",value=4.0)
        x3 = 3.0
        bool_x1_le_x1 = (x1 <= x1)
        bool_x1_le_x2 = (x1 <= x2)
        bool_x1_le_x3 = (x1 <= x3)
        bool_x2_le_x1 = (x2 <= x1)
        bool_x2_le_x3 = (x2 <= x3)
        self.assertEqual(bool_x1_le_x1,True)
        self.assertEqual(bool_x1_le_x2,True)
        self.assertEqual(bool_x1_le_x3,True)
        self.assertEqual(bool_x2_le_x1,False)
        self.assertEqual(bool_x2_le_x3,False)
        sys.modules.pop("pysimplead")

    def test_gt(self):
        from pysimplead import PySAD
        x1 = PySAD(name="x",value=2.0,maxvar=2)
        x2 = PySAD(name="y",value=4.0)
        x3 = 3.0
        bool_x1_gt_x1 = (x1 > x1)
        bool_x1_gt_x2 = (x1 > x2)
        bool_x1_gt_x3 = (x1 > x3)
        bool_x2_gt_x1 = (x2 > x1)
        bool_x2_gt_x3 = (x2 > x3)
        self.assertEqual(bool_x1_gt_x1,False)
        self.assertEqual(bool_x1_gt_x2,False)
        self.assertEqual(bool_x1_gt_x3,False)
        self.assertEqual(bool_x2_gt_x1,True)
        self.assertEqual(bool_x2_gt_x3,True)
        sys.modules.pop("pysimplead")

    def test_ge(self):
        from pysimplead import PySAD
        x1 = PySAD(name="x",value=2.0,maxvar=2)
        x2 = PySAD(name="y",value=4.0)
        x3 = 3.0
        bool_x1_ge_x1 = (x1 >= x1)
        bool_x1_ge_x2 = (x1 >= x2)
        bool_x1_ge_x3 = (x1 >= x3)
        bool_x2_ge_x1 = (x2 >= x1)
        bool_x2_ge_x3 = (x2 >= x3)
        self.assertEqual(bool_x1_ge_x1,True)
        self.assertEqual(bool_x1_ge_x2,False)
        self.assertEqual(bool_x1_ge_x3,False)
        self.assertEqual(bool_x2_ge_x1,True)
        self.assertEqual(bool_x2_ge_x3,True)
        sys.modules.pop("pysimplead")

    def test_eq(self):
        from pysimplead import PySAD
        x1 = PySAD(name="x",value=2.0,maxvar=3)
        x2 = PySAD(name="y",value=4.0)
        x3 = 3.0
        x4 = PySAD(name="x",value=3.0)
        bool_x1_eq_x1 = (x1 == x1)
        bool_x1_eq_x2 = (x1 == x2)
        bool_x1_eq_x3 = (x1 == x3)
        bool_x2_eq_x1 = (x2 == x1)
        bool_x2_eq_x3 = (x2 == x3)
        bool_x4_eq_x3 = (x4 == x3)
        self.assertEqual(bool_x1_eq_x1,True)
        self.assertEqual(bool_x1_eq_x2,False)
        self.assertEqual(bool_x1_eq_x3,False)
        self.assertEqual(bool_x2_eq_x1,False)
        self.assertEqual(bool_x2_eq_x3,False)
        self.assertEqual(bool_x4_eq_x3,True)
        sys.modules.pop("pysimplead")

    def test_ne(self):
        from pysimplead import PySAD
        x1 = PySAD(name="x",value=2.0,maxvar=3)
        x2 = PySAD(name="y",value=4.0)
        x3 = 3.0
        x4 = PySAD(name="x",value=3.0)
        bool_x1_ne_x1 = (x1 != x1)
        bool_x1_ne_x2 = (x1 != x2)
        bool_x1_ne_x3 = (x1 != x3)
        bool_x2_ne_x1 = (x2 != x1)
        bool_x2_ne_x3 = (x2 != x3)
        bool_x4_ne_x3 = (x4 != x3)
        self.assertEqual(bool_x1_ne_x1,False)
        self.assertEqual(bool_x1_ne_x2,True)
        self.assertEqual(bool_x1_ne_x3,True)
        self.assertEqual(bool_x2_ne_x1,True)
        self.assertEqual(bool_x2_ne_x3,True)
        self.assertEqual(bool_x4_ne_x3,False)
        sys.modules.pop("pysimplead")


if __name__ == '__main__':
    unittest.main()
