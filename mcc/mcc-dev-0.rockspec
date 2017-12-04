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
  "compat53",
  "argparse",
  "csv",
  "lua-cjson",
  "luafilesystem",
  "lustache",
  "serpent",
}

build = {
  type    = "builtin",
  modules = {},
  install = {
    bin = {
      ["mcc"] = "mcc.lua",
    },
  },
}
--[[ luacheck: pop ]]
