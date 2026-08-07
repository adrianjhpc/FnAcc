module fnacc_saxpy_kernel
contains

  subroutine saxpy_prepare(x, y)
    real :: x(:), y(:)

    !$fnacc update device(x)
    !$fnacc update device(y)
  end subroutine

  subroutine saxpy_compute(alpha, x, y)
    real :: alpha
    real :: x(:), y(:)
    integer :: i

    !$fnacc parallel tile(128) pack(x:device, y:device)
    do i = 1, size(y)
      y(i) = alpha * x(i) + y(i)
    end do
  end subroutine

  subroutine saxpy_fetch(y)
    real :: y(:)

    !$fnacc update host(y)
  end subroutine

  subroutine saxpy_release(x, y)
    real :: x(:), y(:)

    !$fnacc release(x, y)
  end subroutine

end module

