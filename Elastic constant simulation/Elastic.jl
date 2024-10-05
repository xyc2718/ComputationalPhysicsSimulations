module Elastic

include("modle.jl")
include("MD.jl")
include("cellmin.jl")
using .Model
using .MD
using .cellmin
global const kb=1.38*10^-23

modules = [Model, MD, cellmin]

for mod in modules
    for name in names(mod, all=true)
        if !(name in (:eval, :include, :using, :import, :export, :module, :end))
            @eval export $name
        end
    end
end
end