# -*- coding: utf-8 -*-


import sys
import platform

print('Path:', sys.path)
print('Platform architecture:', platform.architecture())
print('Machine type:', platform.machine())
print('System network name:', platform.node())
print('Platform information:', platform.platform())
print('Platform processor:', platform.processor())
print('Operating system:', platform.system())
print('OS extra info (release, version, csd/service pack, ptype/OS type):', platform.win32_ver())
print('System info:', platform.uname())
print('Python build no. and date:', platform.python_build())
print('Python compiler:', platform.python_compiler())
print('Python SCM:', platform.python_branch())
print('Python implementation:', platform.python_implementation())
print('Python version:', platform.python_version())
