'''
 File Created: Wed Sep 23 2020
 Author: Peng YUN (pyun@ust.hk)
 Copyright 2018-2020 Peng YUN, RAM-Lab, HKUST
'''
import unittest
import subprocess

class TestGeneratePseudoAnnotation(unittest.TestCase):
    # test_no_pseudo_anno
    def test_no_pseudo_anno(self):
        cmd = "python3 main.py "
        cmd += "--tag unittest-TestGeneratePseudoAnnotation-test_no_pseudo_anno "
        cmd += "--cfg-path unit_tests/data/test_pseudo_anno_nopseudoanno.py "
        cmd += "--mode train"
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        self.assertFalse("Generate Pseudo Annotations START" in str(output))
    def test_pseudo_anno(self):
        cmd = "python3 main.py "
        cmd += "--tag unittest-TestGeneratePseudoAnnotation-test_no_pseudo_anno "
        cmd += "--cfg-path unit_tests/data/test_pseudo_anno_pseudoanno.py "
        cmd += "--mode train"
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        self.assertTrue("Generate Pseudo Annotations START" in str(output))
        self.assertFalse("Generate Pseudo Annotations End" in str(output))

if __name__ == "__main__":
    unittest.main()
