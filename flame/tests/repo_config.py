import os

if os.environ.get("TRAVIS_OS_NAME") == "osx":
    MODEL_REPOSITORY = "/Home/travis/testmodels"
else:
    MODEL_REPOSITORY = "/home/testmodels"