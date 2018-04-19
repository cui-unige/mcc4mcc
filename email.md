Dear Model Checking Contest competitors,

This year, the SMV team at University of Geneva will propose a tool
named `mcc4mcc` (model checker collection for the model checking contest)
in the competition, that acts as a wrapper above the tools you submitted
in 2017.

In order to select one of your tools for each examination/instance,
`mcc4mcc` uses machine learning techniques on a new feature available this
year in the contest: the
[generic properties](https://mcc.lip6.fr/developers.php).

The code of `mcc4mcc` is open source (MIT license) and available
[on GitHub](https://github.com/cui-unige/mcc4mcc).
A tool paper has also been accepted at the Petri nets conference on this topic,
put as attachment to this email.

In order to create `mcc4mcc`, we have wrapped all your tools within docker
images, either by compiling the required sources in an
[Alpine Linux](https://www.alpinelinux.org) distribution,
or by copying the binary from your virtual machines.
For each tool, you can look at the `Dockerfile`s and the `prepare` script
that are used to build the images. `prepare` scripts extract required files
from the virtual machines you provided last year.

Currently, all the docker images are stored in a private repository,
but public docker images could be great for your tools.
Feel free to reuse our `Dockerfile`s to build your own docker images.
We would also be happy to work with you on creating the images, as we propose
in the paper a layout for docker images that could be useful for both your
tools and the contest.

Alas, we had to update some of your scripts, and thus to store them in
the `mcc4mcc` repository. Most of the time, it is the `BenchKit_head.sh`
script, that is stored as `mcc-head` in the repository.
For some tools, we also head to extract some other files.
As we do not own intellectual property on these, we need you to explicitly
accept this. If you want the files to be removed, please ask us as soon
as possible and we will perform the deletion.

The model checking contest rules also require us to get your explicit
approval, even if your tool is free software, to let us use it in the
competition. If you do not want your tool to be used within `mcc4mcc`,
ask us as soon as possible to remove it. In the other case,
please send us (with copy to the contest organizers) a small sentence of
approval.

The use of other tools to compete in the model checking contest raises
a problem concerning the fairness of the results of the competition.
The contest commitee is already aware and discussing about this.
Our goal is mainly to provide a base tool to the MCC community, and let you
reuse and improve it. Thus do not worry about the final score: everything will
be made by the organizers to avoid biased results.

Best regards,
Alban Linard
Semantics Modeling & Verification Team
University of Geneva


Note to the developers of smart: we tried to ask a license of smart but did
not obtain an answer, thus smart is currently not included into `mcc4mcc`.
We would be happy to include it, so feel free to contact us!
