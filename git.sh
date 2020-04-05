#!/bin/bash

git fetch origin master
git merge origin master
git add -A
git commit -m "commit"
git push origin master

