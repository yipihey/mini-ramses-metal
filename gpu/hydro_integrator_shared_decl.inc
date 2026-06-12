    type(subgrid_3x2x2cell_primitive), shared :: left_interfaces_x, right_interfaces_x
    type(subgrid_2x3x2cell_primitive), shared :: left_interfaces_y, right_interfaces_y
    type(subgrid_2x2x3cell_primitive), shared :: left_interfaces_z, right_interfaces_z
    type(subgrid_6x6x6cell_primitive), shared :: local_subgrid
#if NVAR>5+NENER
    real(kind=dp), shared :: scalar_subgrid(0:2*nsubgrid+3, 0:2*nsubgrid+3, 0:2*nsubgrid+3, 1:nscalar)
#endif
#ifdef MHD
    type(subgrid_face_b_t), shared :: bf
    type(subgrid_2x3x3edge_primitive), shared :: emfx
    type(subgrid_3x2x3edge_primitive), shared :: emfy
    type(subgrid_3x3x2edge_primitive), shared :: emfz
    type(subgrid_2x3x3edge_corner), shared :: corner_x
    type(subgrid_3x2x3edge_corner), shared :: corner_y
    type(subgrid_3x3x2edge_corner), shared :: corner_z
#endif
