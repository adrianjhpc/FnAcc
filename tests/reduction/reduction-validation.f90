module fnacc_reduction_validation_kernels
  use, intrinsic :: iso_fortran_env, only: real32, real64
  implicit none

contains

  subroutine sum_f32(a, n, result)
    real(real32), intent(in) :: a(:)
    integer, intent(in) :: n
    real(real32), intent(out) :: result
    integer :: i

    result = 0.0_real32
    !$fnacc parallel tile(256) reduction(+:result)
    do i = 1, n
      result = result + a(i)
    end do
  end subroutine sum_f32

  subroutine dot_f32(a, b, n, result)
    real(real32), intent(in) :: a(:), b(:)
    integer, intent(in) :: n
    real(real32), intent(out) :: result
    integer :: i

    result = 0.0_real32
    !$fnacc parallel tile(256) reduction(+:result)
    do i = 1, n
      result = result + a(i) * b(i)
    end do
  end subroutine dot_f32

  subroutine product_f32(a, n, result)
    real(real32), intent(in) :: a(:)
    integer, intent(in) :: n
    real(real32), intent(out) :: result
    integer :: i

    result = 1.0_real32
    !$fnacc parallel tile(256) reduction(*:result)
    do i = 1, n
      result = result * a(i)
    end do
  end subroutine product_f32

  subroutine min_f32(a, n, result)
    real(real32), intent(in) :: a(:)
    integer, intent(in) :: n
    real(real32), intent(out) :: result
    integer :: i

    result = huge(result)
    !$fnacc parallel tile(256) reduction(min:result)
    do i = 1, n
      result = min(result, a(i))
    end do
  end subroutine min_f32

  subroutine max_f32(a, n, result)
    real(real32), intent(in) :: a(:)
    integer, intent(in) :: n
    real(real32), intent(out) :: result
    integer :: i

    result = -huge(result)
    !$fnacc parallel tile(256) reduction(max:result)
    do i = 1, n
      result = max(result, a(i))
    end do
  end subroutine max_f32

  subroutine sum_f64(a, n, result)
    real(real64), intent(in) :: a(:)
    integer, intent(in) :: n
    real(real64), intent(out) :: result
    integer :: i

    result = 0.0_real64
    !$fnacc parallel tile(256) reduction(+:result)
    do i = 1, n
      result = result + a(i)
    end do
  end subroutine sum_f64

  subroutine dot_f64(a, b, n, result)
    real(real64), intent(in) :: a(:), b(:)
    integer, intent(in) :: n
    real(real64), intent(out) :: result
    integer :: i

    result = 0.0_real64
    !$fnacc parallel tile(256) reduction(+:result)
    do i = 1, n
      result = result + a(i) * b(i)
    end do
  end subroutine dot_f64

  subroutine product_f64(a, n, result)
    real(real64), intent(in) :: a(:)
    integer, intent(in) :: n
    real(real64), intent(out) :: result
    integer :: i

    result = 1.0_real64
    !$fnacc parallel tile(256) reduction(*:result)
    do i = 1, n
      result = result * a(i)
    end do
  end subroutine product_f64

  subroutine min_f64(a, n, result)
    real(real64), intent(in) :: a(:)
    integer, intent(in) :: n
    real(real64), intent(out) :: result
    integer :: i

    result = huge(result)
    !$fnacc parallel tile(256) reduction(min:result)
    do i = 1, n
      result = min(result, a(i))
    end do
  end subroutine min_f64

  subroutine max_f64(a, n, result)
    real(real64), intent(in) :: a(:)
    integer, intent(in) :: n
    real(real64), intent(out) :: result
    integer :: i

    result = -huge(result)
    !$fnacc parallel tile(256) reduction(max:result)
    do i = 1, n
      result = max(result, a(i))
    end do
  end subroutine max_f64

end module fnacc_reduction_validation_kernels

program fnacc_reduction_validation
  use, intrinsic :: iso_c_binding, only: c_int64_t
  use, intrinsic :: iso_fortran_env, only: real32, real64
  use fnacc_reduction_validation_kernels
  implicit none

  integer, parameter :: cases(15) = [ &
      1, 2, 3, 31, 32, 33, 127, 128, 129, 255, 256, 257, &
      1000, 65537, 131071 ]

  real(real32), allocatable :: a32(:), b32(:), product32(:)
  real(real64), allocatable :: a64(:), b64(:), product64(:)
  real(real32) :: actual32, expected32, tolerance32
  real(real64) :: actual64, expected64, tolerance64
  integer :: i, n, failures

  integer(c_int64_t) :: primary_launches, stage_launches
  integer(c_int64_t) :: partial_allocations, partial_growths
  integer(c_int64_t) :: partial_reuses, partial_capacity_bytes
  integer(c_int64_t) :: scratch_allocations, scratch_growths
  integer(c_int64_t) :: scratch_reuses, scratch_capacity_bytes

  interface
    subroutine get_workspace_stats(primary_launches, stage_launches, &
        partial_allocations, partial_growths, partial_reuses, &
        partial_capacity_bytes, scratch_allocations, scratch_growths, &
        scratch_reuses, scratch_capacity_bytes) &
        bind(C, name="__fnacc_get_reduction_workspace_stats_v1")
      import :: c_int64_t
      integer(c_int64_t), intent(out) :: primary_launches, stage_launches
      integer(c_int64_t), intent(out) :: partial_allocations, partial_growths
      integer(c_int64_t), intent(out) :: partial_reuses
      integer(c_int64_t), intent(out) :: partial_capacity_bytes
      integer(c_int64_t), intent(out) :: scratch_allocations, scratch_growths
      integer(c_int64_t), intent(out) :: scratch_reuses
      integer(c_int64_t), intent(out) :: scratch_capacity_bytes
    end subroutine get_workspace_stats
  end interface

  allocate(a32(maxval(cases)), b32(maxval(cases)), product32(maxval(cases)))
  allocate(a64(maxval(cases)), b64(maxval(cases)), product64(maxval(cases)))

  do i = 1, maxval(cases)
    a32(i) = real(mod(i, 17) - 8, real32) * 0.125_real32
    b32(i) = real(mod(3 * i, 19) - 9, real32) * 0.0625_real32
    a64(i) = real(mod(i, 17) - 8, real64) * 0.125_real64
    b64(i) = real(mod(3 * i, 19) - 9, real64) * 0.0625_real64
    product32(i) = merge(-1.0_real32, 1.0_real32, mod(i, 7) == 0)
    product64(i) = merge(-1.0_real64, 1.0_real64, mod(i, 7) == 0)
  end do

  failures = 0

  do i = 1, size(cases)
    n = cases(i)

    expected32 = host_sum_f32(a32, n)
    call sum_f32(a32, n, actual32)
    tolerance32 = 64.0_real32 * epsilon(1.0_real32) * &
        max(1.0_real32, abs(expected32))
    call check_f32("sum", n, actual32, expected32, tolerance32, failures)

    expected32 = host_dot_f32(a32, b32, n)
    call dot_f32(a32, b32, n, actual32)
    tolerance32 = 64.0_real32 * epsilon(1.0_real32) * &
        max(1.0_real32, abs(expected32))
    call check_f32("dot", n, actual32, expected32, tolerance32, failures)

    expected32 = host_product_f32(product32, n)
    call product_f32(product32, n, actual32)
    tolerance32 = 4.0_real32 * epsilon(1.0_real32)
    call check_f32("product", n, actual32, expected32, tolerance32, failures)

    expected32 = minval(a32(1:n))
    call min_f32(a32, n, actual32)
    call check_f32("min", n, actual32, expected32, 0.0_real32, failures)

    expected32 = maxval(a32(1:n))
    call max_f32(a32, n, actual32)
    call check_f32("max", n, actual32, expected32, 0.0_real32, failures)

    expected64 = host_sum_f64(a64, n)
    call sum_f64(a64, n, actual64)
    tolerance64 = 128.0_real64 * epsilon(1.0_real64) * &
        max(1.0_real64, abs(expected64))
    call check_f64("sum", n, actual64, expected64, tolerance64, failures)

    expected64 = host_dot_f64(a64, b64, n)
    call dot_f64(a64, b64, n, actual64)
    tolerance64 = 128.0_real64 * epsilon(1.0_real64) * &
        max(1.0_real64, abs(expected64))
    call check_f64("dot", n, actual64, expected64, tolerance64, failures)

    expected64 = host_product_f64(product64, n)
    call product_f64(product64, n, actual64)
    tolerance64 = 4.0_real64 * epsilon(1.0_real64)
    call check_f64("product", n, actual64, expected64, tolerance64, failures)

    expected64 = minval(a64(1:n))
    call min_f64(a64, n, actual64)
    call check_f64("min", n, actual64, expected64, 0.0_real64, failures)

    expected64 = maxval(a64(1:n))
    call max_f64(a64, n, actual64)
    call check_f64("max", n, actual64, expected64, 0.0_real64, failures)
  end do

  call get_workspace_stats(primary_launches, stage_launches, &
      partial_allocations, partial_growths, partial_reuses, &
      partial_capacity_bytes, scratch_allocations, scratch_growths, &
      scratch_reuses, scratch_capacity_bytes)

  call require_stat("primary launches", &
      primary_launches == int(10 * size(cases), c_int64_t), &
      primary_launches, failures)
  call require_stat("hierarchical stage launches", stage_launches > 0, &
      stage_launches, failures)
  call require_stat("partial allocation count", partial_allocations > 0, &
      partial_allocations, failures)
  call require_stat("partial growth count", partial_growths > 0, &
      partial_growths, failures)
  call require_stat("partial reuse count", partial_reuses > 0, &
      partial_reuses, failures)
  call require_stat("partial capacity", partial_capacity_bytes > 0, &
      partial_capacity_bytes, failures)
  call require_stat("scratch allocation count", scratch_allocations > 0, &
      scratch_allocations, failures)
  call require_stat("scratch growth count", scratch_growths > 0, &
      scratch_growths, failures)
  call require_stat("scratch reuse count", scratch_reuses > 0, &
      scratch_reuses, failures)
  call require_stat("scratch capacity", scratch_capacity_bytes > 0, &
      scratch_capacity_bytes, failures)

  if (failures /= 0) then
    write(*, '(a,i0)') "FNACC reduction validation: FAIL, failures=", failures
    error stop 1
  end if

  write(*, '(a)') "FNACC reduction validation: PASS"
  write(*, '(a,i0,a,i0)') "  launches: primary=", primary_launches, &
      " stage=", stage_launches
  write(*, '(a,i0,a,i0,a,i0,a,i0)') "  partials: alloc=", &
      partial_allocations, " grow=", partial_growths, " reuse=", &
      partial_reuses, " capacity=", partial_capacity_bytes
  write(*, '(a,i0,a,i0,a,i0,a,i0)') "  scratch:  alloc=", &
      scratch_allocations, " grow=", scratch_growths, " reuse=", &
      scratch_reuses, " capacity=", scratch_capacity_bytes

contains

  function host_sum_f32(a, n) result(total)
    real(real32), intent(in) :: a(:)
    integer, intent(in) :: n
    real(real32) :: total
    integer :: j

    total = 0.0_real32
    do j = 1, n
      total = total + a(j)
    end do
  end function host_sum_f32

  function host_dot_f32(a, b, n) result(total)
    real(real32), intent(in) :: a(:), b(:)
    integer, intent(in) :: n
    real(real32) :: total
    integer :: j

    total = 0.0_real32
    do j = 1, n
      total = total + a(j) * b(j)
    end do
  end function host_dot_f32

  function host_product_f32(a, n) result(total)
    real(real32), intent(in) :: a(:)
    integer, intent(in) :: n
    real(real32) :: total
    integer :: j

    total = 1.0_real32
    do j = 1, n
      total = total * a(j)
    end do
  end function host_product_f32

  function host_sum_f64(a, n) result(total)
    real(real64), intent(in) :: a(:)
    integer, intent(in) :: n
    real(real64) :: total
    integer :: j

    total = 0.0_real64
    do j = 1, n
      total = total + a(j)
    end do
  end function host_sum_f64

  function host_dot_f64(a, b, n) result(total)
    real(real64), intent(in) :: a(:), b(:)
    integer, intent(in) :: n
    real(real64) :: total
    integer :: j

    total = 0.0_real64
    do j = 1, n
      total = total + a(j) * b(j)
    end do
  end function host_dot_f64

  function host_product_f64(a, n) result(total)
    real(real64), intent(in) :: a(:)
    integer, intent(in) :: n
    real(real64) :: total
    integer :: j

    total = 1.0_real64
    do j = 1, n
      total = total * a(j)
    end do
  end function host_product_f64

  subroutine check_f32(name, n, actual, expected, tolerance, failures)
    character(len=*), intent(in) :: name
    integer, intent(in) :: n
    real(real32), intent(in) :: actual, expected, tolerance
    integer, intent(inout) :: failures

    if (.not. (abs(actual - expected) <= tolerance)) then
      write(*, '(a,a,a,i0,a,es16.8,a,es16.8,a,es16.8)') &
          "f32 ", name, " failed at n=", n, ": actual=", actual, &
          " expected=", expected, " tolerance=", tolerance
      failures = failures + 1
    end if
  end subroutine check_f32

  subroutine check_f64(name, n, actual, expected, tolerance, failures)
    character(len=*), intent(in) :: name
    integer, intent(in) :: n
    real(real64), intent(in) :: actual, expected, tolerance
    integer, intent(inout) :: failures

    if (.not. (abs(actual - expected) <= tolerance)) then
      write(*, '(a,a,a,i0,a,es24.16,a,es24.16,a,es24.16)') &
          "f64 ", name, " failed at n=", n, ": actual=", actual, &
          " expected=", expected, " tolerance=", tolerance
      failures = failures + 1
    end if
  end subroutine check_f64

  subroutine require_stat(name, condition, value, failures)
    character(len=*), intent(in) :: name
    logical, intent(in) :: condition
    integer(c_int64_t), intent(in) :: value
    integer, intent(inout) :: failures

    if (.not. condition) then
      write(*, '(a,a,a,i0)') "workspace statistic failed: ", name, "=", value
      failures = failures + 1
    end if
  end subroutine require_stat

end program fnacc_reduction_validation
