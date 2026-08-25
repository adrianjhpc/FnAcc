module fnacc_daxpy_kernel
contains

  subroutine daxpy_prepare(x, y)
    real(8) :: x(:), y(:)

    !$fnacc update device(x)
    !$fnacc update device(y)
  end subroutine

  subroutine daxpy_compute(alpha, x, y)
    real(8) :: alpha
    real(8) :: x(:), y(:)
    integer :: i

    !$fnacc parallel tile(128) pack(x:device, y:device)
    do i = 1, size(y)
      y(i) = alpha * x(i) + y(i)
    end do
  end subroutine

  subroutine daxpy_fetch(y)
    real(8) :: y(:)

    !$fnacc update host(y)
  end subroutine

  subroutine daxpy_release(x, y)
    real(8) :: x(:), y(:)

    !$fnacc release(x, y)
  end subroutine

end module

