module Elastic

include("modle.jl")
include("MD.jl")
include("cellmin.jl")
# include("mema.jl")
include("deformer.jl")
include("ema.jl")
include("visualize.jl")
using .Model
# using .MEMA
using .MD
using .cellmin
using .Deformer
using .EMA
using .Visualize


modules = [Model, MD, cellmin, Deformer,EMA,Visualize]

for mod in modules
    for name in names(mod, all=true)
        if !(name in (:eval, :include, :using, :import, :export, :module, :end))
            @eval export $name
        end
    end
end
end