abi <abi/4.0>,
include <tunables/global>

profile apptainer /usr/lib/x86_64-linux-gnu/apptainer/bin/starter flags=(unconfined) {
  userns,
  include if exists <local/apptainer>
}
