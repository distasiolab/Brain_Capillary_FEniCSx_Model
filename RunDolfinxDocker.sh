#alias dolfinx='docker run -ti --rm -v $(pwd):/home/fenics/shared -w /home/fenics/shared dolfinx/dolfinx'

docker run -it --rm -v "$(pwd)":/code dolfinx/dolfinx:nightly /bin/bash
