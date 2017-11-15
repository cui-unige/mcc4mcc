--[[ luacheck: push std rockspec ]]
package = "mcc"
version = "dev-0"
source  = {
  url    = "git+https://github.com/saucisson/mcc.git",
  branch = "master",
}

description = {
  summary    = "Model Checker Collection for the Model Checking Contest",
  detailed   = [[]],
  homepage   = "https://github.com/saucisson/mcc",
  license    = "MIT/X11",
  maintainer = "Alban Linard <alban@linard.fr>",
}

dependencies = {
  "lua >= 5.1",
  "argparse",
  "lustache",
  "luafilesystem",
}

build = {
  type    = "builtin",
  modules = {
    ["mcc"] = "mcc.lua",
  },
  install = {
    bin = {
      ["mcc"] = "mcc.lua",
    },
  },
}
--[[ luacheck: pop ]]
