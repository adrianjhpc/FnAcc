module fnacc_axpby_f64_kernel
contains

  subroutine axpby_f64_prepare(a, b, c)
    real(8) :: a(:), b(:), c(:)

    !$fnacc update device(a)
    !$fnacc update device(b)
    !$fnacc update device(c)
  end subroutine

  subroutine axpby_f64_compute(alpha, beta, a, b, c)
    real(8) :: alpha, beta
    real(8) :: a(:), b(:), c(:)
    integer :: i

    !$fnacc parallel tile(128) pack(a:device, b:device, c:device)
    do i = 1, size(c)
      c(i) = alpha * a(i) + beta * b(i)
    end do
  end subroutine

  subroutine axpby_f64_fetch(c)
    real(8) :: c(:)

    !$fnacc update host(c)
  end subroutine

  subroutine axpby_f64_release(a, b, c)
    real(8) :: a(:), b(:), c(:)

    !$fnacc release(a, b, c)
  end subroutine

end module

