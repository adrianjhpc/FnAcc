module fnacc_vector_add_kernel
contains

  subroutine vector_add_prepare(a, b, c)
    real :: a(:), b(:), c(:)

    !$fnacc update device(a)
    !$fnacc update device(b)
    !$fnacc update device(c)
  end subroutine

  subroutine vector_add_compute(a, b, c)
    real :: a(:), b(:), c(:)
    integer :: i

    !$fnacc parallel tile(128) pack(a:device, b:device, c:device)
    do i = 1, size(c)
      c(i) = a(i) + b(i)
    end do
  end subroutine

  subroutine vector_add_fetch(c)
    real :: c(:)

    !$fnacc update host(c)
  end subroutine

  subroutine vector_add_release(a, b, c)
    real :: a(:), b(:), c(:)

    !$fnacc release(a, b, c)
  end subroutine

end module

