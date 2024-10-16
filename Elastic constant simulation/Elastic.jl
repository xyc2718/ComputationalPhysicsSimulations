module Elastic

include("modle.jl")
include("MD.jl")
include("cellmin.jl")
include("mema.jl")

using .Model
using .MEMA
using .MD
using .cellmin


modules = [Model, MD, cellmin, MEMA]

for mod in modules
    for name in names(mod, all=true)
        if !(name in (:eval, :include, :using, :import, :export, :module, :end))
            @eval export $name
        end
    end
end
end