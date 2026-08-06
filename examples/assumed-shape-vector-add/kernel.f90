module fnacc_assumed_shape_kernels
contains

  subroutine vector_add_assumed_shape(a, b, c)
    real :: a(:), b(:), c(:)
    integer :: i
    integer :: n

    n = size(c)

    !$fnacc parallel tile(128) pack(a:device, b:device, c:device)
    do i = 1, n
      c(i) = a(i) + b(i)
    end do

    !$fnacc update host(c)
    !$fnacc release all
  end subroutine

end module

