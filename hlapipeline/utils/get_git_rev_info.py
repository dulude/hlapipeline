#!/bin/env python
"""
Displays revision information on the specified git-controlled repository

Path
----
software/git_helper_scripts/get_git_rev_info.py

Dependencies
------------
None.

Inputs
------
* Required input
    1: *code_path*
        * Path to query for git revision information.

Example
-------
Get git revision info on git repository operations::

    $HLAsoftware/git_helper_scripts/get_git_rev_info.py operations

Functions
---------
None.
"""
import os, sys
#-----------------------------------------------------------------------------------------------------------------------
def print_rev_id(localRepoPath):
	"""
	prints information about the specified local repository to STDOUT. Expected method of execution: command-line or
	shell script call

	:param localRepoPath: local repository path.
	:type localRepoPath: string
	:return: Nothing as such. subroutine will exit with a state of 0 if everything ran OK, and a value of '111' if
	something went wrong.
	"""
	start_path = os.getcwd()
	try:
		os.chdir(localRepoPath)
		print "== Remote URL"
		os.system('git remote -v')

		print "== Remote Branches"
		os.system("git branch -r")

		print "== Local Branches"
		os.system("git branch")

		print "== Configuration (.git/config)"
		os.system("cat .git/config")

		print "== Most Recent Commit"
		os.system("git log |head -1")
	except:
		sys.exit(111)
	finally:
		os.chdir(start_path)
#-----------------------------------------------------------------------------------------------------------------------
def get_rev_id(localRepoPath):
	"""
    returns the current full git revision id of the specified local repository. Expected method of execution: python
    subroutine call

    :param localRepoPath: local repository path.
    :type localRepoPath: string
    :return: full git revision ID of the specified repository if everything ran OK, and "FAILURE" if something went
    wrong.
    """
	start_path = os.getcwd()
	try:
		os.chdir(localRepoPath)

		instream = os.popen("git --no-pager log --max-count=1 | head -1")
		for streamline in instream.readlines():
			streamline = streamline.strip()
			if streamline.startswith("commit "):
				rv = streamline.replace("commit ","")
			else:
				raise
	except:
		rv = "FAILURE: git revision info not found"
	finally:
		os.chdir(start_path)

	return(rv)
#-----------------------------------------------------------------------------------------------------------------------
if(__name__ == '__main__'):
	localRepoPath = sys.argv[1]
	print_rev_id(localRepoPath)
