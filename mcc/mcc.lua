#! /usr/bin/env lua

local Argparse = require "argparse"
local Json     = require "cjson"
local Lfs      = require "lfs"
local Ltn12    = require "ltn12"
local Lustache = require "lustache"
local Http     = require "socket.http"

local parser    = Argparse () {
  name        = "mcc",
  description = "Model Checker Collection for the Model Checking Contest",
  epilog      = "For more info, see https://github.com/saucisson/mcc.",
}
parser:option   "-d" "--docker" {
  description = "docker hostname and port (hostname:port)",
  default     = Lfs.currentdir () == "/mcc-data"
            and "docker:2375"
             or "localhost:2375",
}
parser:option   "-i" "--input" {
  description = "input directory",
  default     = Lfs.currentdir (),
}
parser:argument "examination" {
  description = "examniation type",
  default     = os.getenv "BK_EXAMINATION",
}
local arguments = parser:parse ()
assert (arguments)

os.execute (Lustache:render ([[
  dockerize -wait "tcp://{{{docker}}}"
]], {
  docker = arguments.docker,
}))

local function run (container, tool)
  local _, status
  local service   = {}
  local body = Json.encode {
    Image        = container,
    Volumes      = { ["/mcc-data"] = {} },
    HostConfig   = {
      Binds     = { arguments.input .. ":/mcc-data" },
      LogConfig = { Type = "json-file" },
    },
    AttachStdout = true,
    AttachStderr = true,
    AttachStdin  = false,
    Env          = {
      "BK_TOOL="        .. tool,
      "BK_EXAMINATION=" .. arguments.examination,
    },
  }
  _, status = Http.request {
    url    = Lustache:render ("http://{{{docker}}}/containers/create", {
      docker = arguments.docker,
    }),
    method = "POST",
    headers = {
      ["Content-type"  ] = "application/json",
      ["Content-length"] = #body,
    },
    sink   = Ltn12.sink.table (service),
    source = Ltn12.source.string (body),
  }
  assert (status == 201, status)
  service = Json.decode (table.concat (service))
  _, status = Http.request {
    method = "POST",
    url    = Lustache:render ("http://{{{docker}}}/containers/{{{id}}}/start", {
      docker = arguments.docker,
      id     = service.Id,
    }),
  }
  assert (status == 204, status)
  local finished = {}
  _, status = Http.request {
    method = "POST",
    sink   = Ltn12.sink.table (finished),
    url    = Lustache:render ("http://{{{docker}}}/containers/{{{id}}}/wait", {
      docker = arguments.docker,
      id     = service.Id,
    }),
  }
  assert (status == 200, status)
  finished = Json.decode (table.concat (finished))
  local logs = {}
  _, status = Http.request {
    method = "GET",
    sink   = Ltn12.sink.table (logs),
    url    = Lustache:render ("http://{{{docker}}}/containers/{{{id}}}/logs?stdout=true", {
      docker = arguments.docker,
      id     = service.Id,
    }),
  }
  assert (status == 200, status)
  logs = table.concat (logs)
  io.stdout:write (logs .. "\n")
  _, status = Http.request {
    method = "GET",
    sink   = Ltn12.sink.table (logs),
    url    = Lustache:render ("http://{{{docker}}}/containers/{{{id}}}/logs?stderr=true", {
      docker = arguments.docker,
      id     = service.Id,
    }),
  }
  assert (status == 200, status)
  logs = table.concat (logs)
  io.stderr:write (logs .. "\n")
  assert (finished.StatusCode == 0)
end

local ok, err = pcall (function ()
  -- local modelfile  = read_file "model.pnml"
  -- local is_colored = read_line "iscolored"
  -- local colored    = read_line "equiv_col"
  -- local pt         = read_line "equiv_pt"
  -- local instance   = read_line "instance"
  local container, tool
  if     arguments.examination == "StateSpace"              then
  elseif arguments.examination == "UpperBounds"             then
  elseif arguments.examination == "ReachabilityDeadlock"    then
  elseif arguments.examination == "ReachabilityFireability" then
  elseif arguments.examination == "ReachabilityCardinality" then
  elseif arguments.examination == "LTLFireability"          then
  elseif arguments.examination == "LTLCardinality"          then
  elseif arguments.examination == "CTLFireability"          then
  elseif arguments.examination == "CTLCardinality"          then
  else
    assert (false, "Unknown examination: " .. tostring (examination))
  end
  run (container, tool)
end)

if not ok then
  print "CANNOT COMPUTE"
  io.stderr:write (err .. "\n")
end
