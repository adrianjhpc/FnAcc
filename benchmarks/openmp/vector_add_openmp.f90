program vector_add_openmp
  use bench_utils
  use omp_lib
  implicit none

  integer(8) :: n, i
  integer :: reps, r, errors
  real, allocatable :: a(:), b(:), c(:)
  real :: expected
  real(8) :: t0, t1, elapsed, bytes_per_rep

  call parse_i64_arg(1, 1048576_8, n)
  call parse_i32_arg(2, 100, reps)

  allocate(a(n), b(n), c(n))

  do i = 1, n
    a(i) = real(i)
    b(i) = 2.0 * real(i)
    c(i) = 0.0
  end do

  !$omp parallel do
  do i = 1, n
    c(i) = a(i) + b(i)
  end do


  t0 = wall_time()
  !$omp parallel private(r, i)
  do r = 1, reps
    !$omp  do
    do i = 1, n
      c(i) = a(i) + b(i)
    end do
  end do
  !$omp end parallel
  t1 = wall_time()

  errors = 0
  do i = 1, n
    expected = a(i) + b(i)
    if (.not. almost_equal_f32(c(i), expected)) errors = errors + 1
  end do

  elapsed = t1 - t0
  bytes_per_rep = 3.0d0 * real(n, 8) * 4.0d0

  call print_result("openmp_vector_add", n, 1_8, reps, elapsed, bytes_per_rep, errors)
end program

