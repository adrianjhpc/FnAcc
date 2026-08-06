module fnacc_update_before_launch_kernels
contains

  subroutine update_then_launch(a, b, c)
    real :: a(:), b(:), c(:)
    integer :: i

    !$fnacc update device(a)
    !$fnacc update device(b)

    !$fnacc parallel tile(128) pack(a:device, b:device, c:device)
    do i = 1, size(c)
      c(i) = a(i) + b(i)
    end do

    !$fnacc update host(c)
    !$fnacc release all
  end subroutine

end module

