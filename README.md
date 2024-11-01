# project for the Computational Physics Simulation

* These are projects for the Computational Physics Simulation Laboratory course at Fudan University for the Fall 2024 semester.

###Elastic Constant for Al,Cu and Diamond in 0K and finite temperature
We calculate the elastic properties of Al,Cu and diamond in 0K using structure optimization and in finite temperature using MD simulation.
EMA potential of Al and Cu are implemented in ema.jl 
SW potential for diamond are implemented in sw.jl
Different MD simulation for NPT and NVT are implemented in MD.jl
    include:MD for NPT ensemble:
                        Andersen Nose-Hoover method: Intergrate by 3 order Runge Kuta or multistep of Liouville operator and Tort Decomposition
                        Multistep of Andersen Langvin method
                    NVT ensemble:
                        Nose-Hoover method:Intergrate by 3 order Runge Kuta 
    multi-threaded force computation is implemented on the CPU.
